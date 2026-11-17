# Data, security and privacy: Parcel Locker

## Data collected
The system stores the resident name, phone number, apartment number and the parcel history.
Camera footage is not recorded by the locker.

## Data protection
Data is encrypted in transit with TLS 1.3 and at rest with AES-256. Phone numbers are stored hashed for search and encrypted for display. [TO VERIFY] The hashing algorithm may change to a salted scheme in the next release.

## Application security
The pipeline runs a static analyzer for Python, a dependency vulnerability scanner and a container image scanner on every pull request.
A penetration test is done once a year by an external company.

## Privacy and data retention
Parcel history is kept for 12 months and then anonymized.
A resident can ask for data deletion by e-mail, and the request is answered in 15 days.

## Incident response
Security incidents are triaged within 4 hours. The on-call engineer notifies the security lead, who decides on customer communication.
