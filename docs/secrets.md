# Secrets handling

## Today (local / portfolio)

The app needs one secret, `OPENAI_API_KEY`. It lives in a local `.env` file that is
git-ignored and never committed. Application code does not read `os.environ` directly
for it — it calls `factory_floor.secrets.get_secret("OPENAI_API_KEY")`, which with the
default backend (`FACTORY_FLOOR_SECRETS_BACKEND=env`) is just `os.getenv`.

That indirection is the whole point of this phase: it is the seam a managed vault plugs
into, with **no change to application code**.

## Production (deploy-time)

At deploy time the secret should come from a managed store, not a file on disk:

| Backend value | Store | Where the client call goes |
|---|---|---|
| `aws` | AWS Secrets Manager | `boto3.client("secretsmanager").get_secret_value(...)` |
| `vault` | HashiCorp Vault | `hvac.Client(...).secrets.kv.v2.read_secret_version(...)` |
| `doppler` | Doppler | `doppler secrets download` / the Doppler SDK |
| `sops` | SOPS-encrypted file | decrypt at boot, load into the environment |

The flow is: the CI/CD pipeline (or the container runtime) authenticates to the vault
with its own workload identity, fetches the secret, and exposes it to the process —
typically still as an environment variable, in which case the default `env` backend
keeps working and nothing else changes. `get_secret()` only needs a non-`env` backend
if the process must pull the secret itself at runtime; in that case implement the
matching branch in `factory_floor/secrets.py` (each is currently a `NotImplementedError`
marking the exact spot).

### Checklist for a real deployment

- [ ] Provision the secret in the chosen vault; grant the workload read access.
- [ ] Pick the injection style: env-var injection (keep `env` backend) vs. runtime
      fetch (implement the backend branch).
- [ ] Never bake the key into an image layer or a committed file.
- [ ] Rotate on a schedule; the app picks up a new value on restart.
- [ ] Keep `.env` for local development only; it is already git-ignored.
