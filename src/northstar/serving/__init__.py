"""Section 08 - productionization: versioned model artifacts, a scoring API, batch scoring and
drift/performance monitoring for the section 01 lead score and the section 02 churn model.

Modules: ``specs`` (what is served), ``training`` (offline fit + registration), ``registry``
(artifact storage and verified loading), ``schemas`` (request/response contracts), ``scoring``
(shared scoring path), ``api`` (FastAPI app), ``batch`` (batch scoring), ``profiles`` and
``monitoring`` (drift and performance checks), ``report`` (section 08 outputs).
"""
