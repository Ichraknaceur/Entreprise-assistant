# Backup and Restore Policy

## Purpose
This policy defines backup and restore rules for business-critical systems and data.

## Backup Frequency
Production databases are backed up daily. Critical configuration data and shared file repositories are backed up according to system-specific schedules.

## Retention
Daily backups are retained for 30 days. Monthly backups are retained for 12 months unless regulation requires longer retention.

## Encryption
All backups must be encrypted both in transit and at rest.

## Restore Testing
Restore procedures must be tested at least quarterly to verify recovery integrity and operational readiness.

## Responsibilities
System owners are responsible for ensuring their applications are covered by approved backup procedures. Infrastructure teams manage backup execution and monitoring.

## Exceptions
Any exception to this policy must be documented and approved by IT governance.
