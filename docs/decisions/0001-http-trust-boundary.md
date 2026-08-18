# ADR 0001: Keep the OCR API on an explicit trusted boundary

Status: accepted for the current deployment contract.

## Context

The HTTP API accepts PDFs, exposes durable job metadata/results, and performs
expensive GPU work. It currently has no authentication or tenant authorization.
The CLI binds to `127.0.0.1` by default. Docker listens on all interfaces inside
the container, but the base Compose file publishes only to host
`127.0.0.1`. The optional shared-network override exposes the service to members
of an explicitly chosen Docker network.

## Decision

Treat the service as trusted-local/trusted-network-only. Keep the default host
publication on localhost. Use the shared Docker network only when every attached
container is trusted to submit jobs and read their status/results.

Do not add application authentication until there is a concrete remote or
multi-tenant consumer with an identity, credential distribution, rotation, and
authorization model. Authentication added without those decisions would not
solve job/result isolation.

## Consequences

- Never publish port 8000 on a non-loopback host address or route it through a
  public reverse proxy under the current contract.
- Treat membership in the optional shared Docker network as privileged access.
- Keep absolute-path and diagnostic disclosure hardening in the backlog; a
  trusted boundary reduces exposure but does not make path leakage desirable.
- Before any remote deployment, add request/resource limits, authenticated
  principals, per-job authorization tests, TLS termination, and audit logging.

Rollback is documentation-only: supersede this ADR when an authenticated
deployment design and consumer are accepted.
