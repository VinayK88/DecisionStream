# Security Policy

## Supported version

Security fixes are applied to the latest commit on `main`.

## Reporting a vulnerability

Please do not disclose suspected vulnerabilities in a public issue. Use GitHub's private vulnerability reporting option from the repository **Security** tab when available. Otherwise, contact the maintainer through the GitHub profile linked from this repository.

Include the affected component, reproduction steps, potential impact, and suggested mitigation. Do not include credentials, production data, or information obtained without authorization.

## Project boundary

DecisionStream is a synthetic reference implementation. It is not connected to a live event bus, customer data, or enforcement plane. Production use requires authenticated ingestion, privacy controls, event-time correctness, feature-store governance, monitoring, and rollback.
