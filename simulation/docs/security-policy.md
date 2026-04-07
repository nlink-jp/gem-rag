# Security Policy

## Authentication

All API access requires Application Default Credentials (ADC).
Service accounts must have the minimum required IAM roles.
Token rotation is enforced every 90 days.

## Incident Response

When a security incident is detected:
1. The SOC team initiates triage within 15 minutes
2. Escalation to CSIRT occurs if the incident severity is HIGH or CRITICAL
3. The incident commander coordinates containment and remediation
4. A post-incident review is conducted within 5 business days

## Data Classification

| Level | Description | Example |
|-------|-------------|---------|
| Public | No restrictions | Marketing materials |
| Internal | Company-wide access | Internal wiki pages |
| Confidential | Need-to-know basis | Customer PII, financial data |
| Restricted | Named individuals only | Encryption keys, audit logs |

## Access Control

Role-based access control (RBAC) is enforced across all systems.
Privileged access requires MFA and is logged to the audit trail.
Quarterly access reviews are mandatory for all Confidential and Restricted resources.
