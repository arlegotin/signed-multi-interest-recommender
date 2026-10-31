"""Small recoverable publication transactions for saved aggregate exports.

Each transaction first retains a validated bundle of new files and byte-exact
backups on the publication filesystem. A durable manifest precedes projection
replacement; the selector is replaced last. Ledger-owned publication becomes
committed only after its exact successful terminal output. Recovery rolls back
unfinished owners, or finishes projecting a durably completed owner's bundle.
Readers must verify the selector's receipt and terminal owner before treating a
publication as successful. Bundles and prior selectors remain available.
"""
import json
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from .budget import atomic_write_json
from .protocol import canonical_sha256
from .stage_evidence import file_reference, verify_record, verify_reference


def _sync_directory(path):
    descriptor=os.open(path,os.O_RDONLY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _replace_from(source,target):
    """Copy a retained bundle member to a projection with durable replacement."""
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    descriptor,temporary=tempfile.mkstemp(prefix=target.name+'.',suffix='.publication',dir=target.parent)
    try:
        with os.fdopen(descriptor,'wb') as handle,Path(source).open('rb') as original:
            shutil.copyfileobj(original,handle);handle.flush();os.fsync(handle.fileno())
        os.replace(temporary,target);_sync_directory(target.parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _seal(value):
    value=dict(value,schema_version=1)
    value['output_sha256']=canonical_sha256(value)
    return value


def _completed(transaction,ledger):
    if transaction.get('owner') is None:return transaction.get('phase')=='committed'
    if ledger is None:raise ValueError('Interrupted publication requires its ledger for recovery')
    attempt=ledger.snapshot().get('attempts',{}).get(transaction['owner'],{})
    return (attempt.get('event')=='finish' and attempt.get('stage')=='G6'
            and attempt.get('status')=='completed' and attempt.get('outputs')==transaction['outputs'])


def _project(files,which):
    # Validate every retained member before changing even one projection.
    for row in files:
        if row[which] is not None:verify_reference(row[which])
    for row in files:
        path=Path(row['target']);reference=row[which]
        if reference is None:
            if path.exists():path.unlink();_sync_directory(path.parent)
        elif not path.exists() or file_reference(path)['sha256']!=reference['sha256']:
            _replace_from(Path(reference['path']),path)


@contextmanager
def _publication_lock(root):
    """Serialize projection/recovery across both library and native writers."""
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    with (root/'.aggregate-publication.lock').open('a') as handle:
        try:fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:raise ValueError('Another aggregate publication writer is active') from error
        try:yield
        finally:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def _recover_publication(root,name,*,ledger=None,allowed_roots=None):
    """Recover a pending transaction; refuse corrupted or out-of-scope records."""
    root=Path(root).resolve();transaction_path=root/f'.{name}.publication.json'
    if not transaction_path.exists():return None
    transaction=json.loads(transaction_path.read_text());verify_record(transaction)
    roots=[Path(p).resolve() for p in (allowed_roots or [root])]
    bundle=Path(transaction['bundle']).resolve()
    if (transaction.get('name')!=name or bundle.parent!=root/'.publications'
            or transaction.get('phase') not in ('prepared','committed') or not transaction.get('files')):
        raise ValueError('Invalid publication transaction identity')
    targets=[Path(row['target']).resolve() for row in transaction['files']]
    if len(set(targets))!=len(targets) or any(not any(p.is_relative_to(r) for r in roots) for p in targets):
        raise ValueError('Publication transaction target escapes its declared scope')
    for row in transaction['files']:
        for kind in ('old','new'):
            if row[kind] is not None and not Path(row[kind]['path']).resolve().is_relative_to(bundle):
                raise ValueError('Publication retained member escapes its bundle')
    committed=_completed(transaction,ledger)
    _project(transaction['files'],'new' if committed else 'old')
    transaction_path.unlink();_sync_directory(root)
    return 'committed' if committed else 'restored'


def _publish_files(root,name,files,selector,selector_record,*,ledger=None,owner=None,outputs=None,allowed_roots=None):
    """Publish validated staged files, retaining a recoverable previous bundle.

    ``files`` maps final paths to staged files (or None for an intentional
    renderer-owned deletion). Stage all files before calling. The selector is
    projection metadata; its contents must identify the validated receipt.
    A terminal-accounting error restores the old publication unless the exact
    completed G6 output was already durably appended.
    """
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    roots=[Path(p).resolve() for p in (allowed_roots or [root])]
    _recover_publication(root,name,ledger=ledger,allowed_roots=roots)
    selector=Path(selector).resolve()
    targets={Path(target).resolve():None if source is None else Path(source).resolve() for target,source in files.items()}
    if selector in targets:raise ValueError('Selector must be published exactly once')
    if owner is not None and ledger is None:raise ValueError('Publication owner requires a ledger')
    for path in [*targets,selector]:
        if not any(path.is_relative_to(r) for r in roots):raise ValueError('Publication target outside declared roots')
        ancestor=path.parent
        while not ancestor.exists():ancestor=ancestor.parent
        if ancestor.stat().st_dev!=root.stat().st_dev:raise ValueError('Publication requires one filesystem')
    bundle=root/'.publications'/f'{name}-{uuid.uuid4().hex}'
    (bundle/'new').mkdir(parents=True);(bundle/'old').mkdir()
    selector_source=bundle/'selector.json';atomic_write_json(selector_source,selector_record)
    targets[selector]=selector_source
    rows=[]
    for index,(target,source) in enumerate(targets.items()):
        row=dict(target=str(target),old=None,new=None)
        for kind,original in (('old',target if target.exists() else None),('new',source)):
            if original is None:continue
            destination=bundle/kind/str(index)
            _replace_from(original,destination)
            row[kind]=file_reference(destination)
        rows.append(row)
    transaction=_seal(dict(name=name,bundle=str(bundle),phase='prepared',files=rows,
                            owner=owner,outputs=outputs or {}))
    # The immutable bundle manifest makes both selected and previous bytes
    # independently recoverable even after the pending transaction is cleared.
    atomic_write_json(bundle/'manifest.json',transaction)
    # The manifest flushes links inside the bundle. Flush its own directory
    # entry and any newly created .publications link before publishing a pending
    # transaction that could otherwise survive a power loss without its backups.
    _sync_directory(bundle.parent)
    _sync_directory(root)
    transaction_path=root/f'.{name}.publication.json'
    atomic_write_json(transaction_path,transaction)
    try:
        _project(rows,'new')
        if owner is not None:
            ledger.finish(owner,'completed',{},outputs or {})
            if not _completed(transaction,ledger):raise ValueError('Publication owner did not complete successfully')
        transaction.pop('output_sha256');transaction['phase']='committed'
        transaction=_seal(transaction);atomic_write_json(transaction_path,transaction)
    except BaseException:
        # If restoration itself fails, leave the durable manifest and backups
        # intact. The next explicit recovery diagnoses/replays that transaction.
        _recover_publication(root,name,ledger=ledger,allowed_roots=roots)
        raise
    _recover_publication(root,name,ledger=ledger,allowed_roots=roots)
    return bundle


def recover_publication(root,name,*,ledger=None,allowed_roots=None):
    """Recover under the same single-writer lock used for publication."""
    with _publication_lock(root):
        return _recover_publication(root,name,ledger=ledger,allowed_roots=allowed_roots)


def publish_files(root,name,files,selector,selector_record,*,ledger=None,owner=None,outputs=None,allowed_roots=None):
    """Publish a validated bundle while excluding competing projection writers."""
    with _publication_lock(root):
        return _publish_files(root,name,files,selector,selector_record,ledger=ledger,owner=owner,
                              outputs=outputs,allowed_roots=allowed_roots)
