# Security Policy

PhysAI is scientific software that evaluates user supplied equations, residuals, callbacks, and geometry functions. Please report security issues privately so that maintainers can investigate them before details are made public.

## Supported versions

Security fixes are made against the latest stable release. Users should upgrade to the latest stable release before reporting an issue when practical. The project does not currently publish a separate long term support schedule.

## Reporting a vulnerability

Send a report to **mankrit.singh.physics@gmail.com** with **"PhysAI security report"** in the subject. This address is listed as the project maintainer contact in `pyproject.toml` and `CITATION.cff`. Please do not open a public issue or pull request for an unpatched vulnerability.

Include as much of the following as you can safely share:

- The affected PhysAI version, Python version, operating system, and relevant backend or dependency versions.
- A concise description of the vulnerability and its potential impact.
- Reproduction steps or a minimal proof of concept, with secrets and personal data removed.
- Any known mitigations or whether the issue is being actively exploited.

You should receive an acknowledgment within five business days. The maintainer will work with you to validate the report, assess affected releases, and coordinate a fix and disclosure timeline. Please allow time for a fix to be prepared before publishing technical details. If you do not receive an acknowledgment, follow up using the same address.

Please do not include credentials, private datasets, or other sensitive material in an initial report. Share only the minimum information needed to reproduce the issue.

## Scope

Reports are welcome for vulnerabilities in PhysAI itself, including unsafe handling of inputs, unintended code execution, exposure of data, or security-relevant failures in supported workflows. Reports about third-party dependencies should be sent to their maintainers as well; please explain any specific impact on PhysAI when reporting them here.

This policy does not treat numerical inaccuracies, expected resource use from deliberately large workloads, or unsupported configurations as security vulnerabilities unless they create a security impact.
