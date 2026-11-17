# Software: Parcel Locker

## Architecture and stack
The backend is a Python service built with FastAPI and SQLAlchemy. The mobile app is built with React Native.
The locker runs a kiosk application that talks to the backend over HTTPS.

## Database
Production uses PostgreSQL 15. Redis is used for short-lived pickup codes and rate limiting.

## Cloud and infrastructure
Services run on a managed container platform in two regions: Frankfurt (primary) and Virginia (secondary).
Backups of the database run every night and are kept for 30 days.

## CI/CD and development cycle
The team uses GitHub Actions. Every pull request runs lint and unit tests. Merging to main deploys to staging automatically, and production deploys need a manual approval.
Work is organized in two-week sprints with a weekly planning meeting.

## Offline behavior
If the locker loses connectivity, it keeps opening compartments for codes issued in the last 24 hours. [TO VERIFY] The offline window may be extended to 48 hours.

## Integrations
Courier systems integrate through a REST API with API keys. Webhooks notify couriers when a parcel is collected.
