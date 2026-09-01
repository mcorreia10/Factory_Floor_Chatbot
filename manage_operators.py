"""Pequeno utilitário para gerir os operadores que podem entrar na app.

Os PINs NUNCA são guardados em texto — só um hash (PBKDF2) fica no operators.csv.
Por isso não se acrescenta/edita uma linha à mão; usa-se este script.

Uso:
    python manage_operators.py list
    python manage_operators.py add  OP-3001 "Marcelo Correia" supervisor 2468
    python manage_operators.py setpin OP-1001 9999
    python manage_operators.py remove OP-3001

Depois de qualquer alteração, reinicia a app (a config/lista é lida no arranque).
"""

import csv
import sys
from pathlib import Path

from factory_floor import identity
from factory_floor.config import OPERATORS_CSV

FIELDS = identity.CSV_FIELDS  # operator_id,name,role,tenant_id,salt,pin_hash
ROLES = ("technician", "supervisor")


def _read():
    p = Path(OPERATORS_CSV)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(rows):
    with Path(OPERATORS_CSV).open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def cmd_list():
    rows = _read()
    if not rows:
        print("(sem operadores)")
        return
    print(f"{'ID':<10} {'Nome':<22} {'Papel':<12} Tenant")
    for r in rows:
        print(f"{r['operator_id']:<10} {r['name']:<22} {r['role']:<12} {r['tenant_id']}")
    print("\n(os PINs não são visíveis — só o hash é guardado)")


def cmd_add(operator_id, name, role, pin, tenant_id="default"):
    if role not in ROLES:
        sys.exit(f"papel tem de ser um de: {', '.join(ROLES)}")
    rows = _read()
    if any(r["operator_id"] == operator_id for r in rows):
        sys.exit(f"{operator_id} já existe — usa 'setpin' para mudar o PIN, ou 'remove' primeiro")
    salt = identity.new_salt()
    rows.append({
        "operator_id": operator_id, "name": name, "role": role,
        "tenant_id": tenant_id, "salt": salt, "pin_hash": identity.hash_pin(pin, salt),
    })
    _write(rows)
    print(f"adicionado: {operator_id} ({name}, {role}) — PIN {pin}")


def cmd_setpin(operator_id, pin):
    rows = _read()
    for r in rows:
        if r["operator_id"] == operator_id:
            r["salt"] = identity.new_salt()
            r["pin_hash"] = identity.hash_pin(pin, r["salt"])
            _write(rows)
            print(f"PIN de {operator_id} mudado para {pin}")
            return
    sys.exit(f"{operator_id} não encontrado")


def cmd_remove(operator_id):
    rows = _read()
    kept = [r for r in rows if r["operator_id"] != operator_id]
    if len(kept) == len(rows):
        sys.exit(f"{operator_id} não encontrado")
    _write(kept)
    print(f"removido: {operator_id}")


def main(argv):
    if not argv:
        print(__doc__)
        return
    cmd, *args = argv
    try:
        if cmd == "list":
            cmd_list()
        elif cmd == "add":
            cmd_add(*args)
        elif cmd == "setpin":
            cmd_setpin(*args)
        elif cmd == "remove":
            cmd_remove(*args)
        else:
            sys.exit(f"comando desconhecido: {cmd}\n{__doc__}")
    except TypeError:
        sys.exit(f"argumentos errados para '{cmd}'.\n{__doc__}")


if __name__ == "__main__":
    main(sys.argv[1:])
