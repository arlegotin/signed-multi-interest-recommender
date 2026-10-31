#!/usr/bin/env python3
"""Analytic synthetic regressions, not an optimizer audit or interval certificate."""
import json
import numpy as np

from signed_study.models import signed_scores


def audit():
    P=np.array([[1.,0.]])
    N=np.array([[.8,.6],[.8,-.6]])
    theta=np.linspace(0,2*np.pi,513)
    X=np.column_stack((np.cos(theta),np.sin(theta)))
    error=float(np.max(np.abs(signed_scores(X,P,N,1.25)+.75*np.abs(X[:,1]))))
    assert error<1e-14
    simplex=[]
    for h in (1,2,3,8):
        centered=np.eye(h+1)-np.ones((h+1,h+1))/(h+1)
        vertices=centered*np.sqrt((h+1)/h)
        e=np.ones(h+1)/np.sqrt(h+1)
        values=signed_scores(-vertices,e[None,:],vertices,2)
        lower=signed_scores(-vertices,vertices,e[None,:],1)
        np.testing.assert_allclose(values,-2/h,rtol=0,atol=1e-14)
        np.testing.assert_allclose(lower,1/h,rtol=0,atol=1e-14)
        simplex.append(dict(residual_dimension=h,ball_upper=0,sphere_upper=-2/h,
                            swapped_ball_lower=0,swapped_sphere_lower=1/h))
    positive=np.array([[np.sqrt(3)/2,.5],[-np.sqrt(3)/2,.5]])
    negative=np.array([[.2,2*np.sqrt(6)/5]])
    candidates=np.array([[0.,1.],[1.,0.]])
    def centroid_score(X,pos,neg):
        prototype=pos.mean(axis=0);prototype/=np.linalg.norm(prototype)
        return signed_scores(X,prototype[None,:],neg,1)
    def lift(X):return np.column_stack((X/np.sqrt(2),np.full(len(X),1/np.sqrt(2))))
    original=centroid_score(candidates,positive,negative)
    lifted=centroid_score(lift(candidates),lift(positive),lift(negative))
    assert original[0]>original[1] and lifted[0]<lifted[1]
    return dict(schema_version=1,cancellation_max_error=error,simplex=simplex,
        original_scores=original.tolist(),lifted_scores=lifted.tolist(),
        scope='analytic examples with finite floating-point checks; no general optimizer or certificate')


if __name__=='__main__':
    print(json.dumps(audit(),sort_keys=True,indent=2,allow_nan=False))
