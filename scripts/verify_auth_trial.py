"""Verification test for 1-Day Trial Access, Guest Tokens, and Master Auth."""

import base64
import sys
from fastapi.testclient import TestClient

from quantedge.api.app import app, create_trial_token, verify_trial_token


def pass_check(title: str, detail: str = "") -> None:
    msg = f"[PASS] {title}"
    if detail:
        msg += f": {detail}"
    print(msg)


def fail_check(title: str, detail: str = "") -> None:
    msg = f"[FAIL] {title}"
    if detail:
        msg += f": {detail}"
    print(msg)
    sys.exit(1)


def main():
    print("==================================================================")
    print("  VERIFYING AUTHENTICATION & 1-DAY TRIAL ACCESS SYSTEM            ")
    print("==================================================================")

    client = TestClient(app)

    # 1. Test public trial pass generation endpoint
    res = client.get("/api/v1/auth/trial-pass")
    if res.status_code != 200:
        fail_check("GET /api/v1/auth/trial-pass status", str(res.status_code))
    data = res.json()
    token = data.get("token")
    if not token or not token.startswith("trial_"):
        fail_check("Trial token structure", str(token))
    pass_check("Public /api/v1/auth/trial-pass issued 24h token", token)

    # 2. Test accessing protected endpoint with 1-day trial token (Basic auth)
    auth_hdr = f"Basic {base64.b64encode(f'guest:{token}'.encode()).decode()}"
    res_protected = client.get("/api/v1/bot/time-limits", headers={"Authorization": auth_hdr})
    if res_protected.status_code != 200:
        fail_check("Access with trial token", f"Status {res_protected.status_code}: {res_protected.text}")
    pass_check("1-Day trial token successfully unlocked protected API (Basic)", "Status 200 OK")

    # 2b. Test accessing protected endpoint with Bearer trial token
    res_bearer = client.get("/api/v1/bot/time-limits", headers={"Authorization": f"Bearer {token}"})
    if res_bearer.status_code != 200:
        fail_check("Access with Bearer trial token", f"Status {res_bearer.status_code}: {res_bearer.text}")
    pass_check("1-Day trial token successfully unlocked protected API (Bearer)", "Status 200 OK")

    # 3. Test Master Password access (Bot@2026, Bot2026, bot2026, and whitespace)
    master_auth = f"Basic {base64.b64encode(b'operator:Bot@2026').decode()}"
    res_master = client.get("/api/v1/bot/time-limits", headers={"Authorization": master_auth})
    if res_master.status_code != 200:
        fail_check("Master password access (Bot@2026)", str(res_master.status_code))
    pass_check("Master password (Bot@2026) verified", "Status 200 OK")

    master_auth2 = f"Basic {base64.b64encode(b'custom_username:Bot2026').decode()}"
    res_master2 = client.get("/api/v1/bot/time-limits", headers={"Authorization": master_auth2})
    if res_master2.status_code != 200:
        fail_check("Master password access (Bot2026 + custom username)", str(res_master2.status_code))
    pass_check("Master password (Bot2026 with any username) verified", "Status 200 OK")

    master_auth_lower = f"Basic {base64.b64encode(b'operator:bot2026').decode()}"
    res_lower = client.get("/api/v1/bot/time-limits", headers={"Authorization": master_auth_lower})
    if res_lower.status_code != 200:
        fail_check("Master password access (lowercase bot2026)", str(res_lower.status_code))
    pass_check("Master password (lowercase bot2026) verified", "Status 200 OK")

    master_auth_space = f"Basic {base64.b64encode(b'  user_space  :  Bot2026   ').decode()}"
    res_space = client.get("/api/v1/bot/time-limits", headers={"Authorization": master_auth_space})
    if res_space.status_code != 200:
        fail_check("Master password access (with whitespace)", str(res_space.status_code))
    pass_check("Master password (with leading/trailing whitespace) verified", "Status 200 OK")

    # 4. Test dedicated trader account with Trader@2026 and Trader2026
    trader_at_auth = f"Basic {base64.b64encode(b'trader:Trader@2026').decode()}"
    res_at = client.get("/api/v1/bot/time-limits", headers={"Authorization": trader_at_auth})
    if res_at.status_code != 200:
        fail_check("Trader account access (trader:Trader@2026)", str(res_at.status_code))
    pass_check("Trader account (trader:Trader@2026) verified", "Status 200 OK")

    secondary_auth = f"Basic {base64.b64encode(b'trader:Trader2026').decode()}"
    res_sec = client.get("/api/v1/bot/time-limits", headers={"Authorization": secondary_auth})
    if res_sec.status_code != 200:
        fail_check("Secondary account access (trader:Trader2026)", str(res_sec.status_code))
    pass_check("Secondary account (trader:Trader2026) verified", "Status 200 OK")

    # 4b. Test that typing username as password fails (e.g. trader:trader MUST NOT log in)
    trader_bad_auth = f"Basic {base64.b64encode(b'trader:trader').decode()}"
    res_bad_trader = client.get("/api/v1/bot/time-limits", headers={"Authorization": trader_bad_auth})
    if res_bad_trader.status_code != 401:
        fail_check("Rejection of username as password (trader:trader)", f"Expected 401, got {res_bad_trader.status_code}")
    pass_check("Username as password (trader:trader) properly rejected with 401", "Unauthorized")

    # 4c. Test dedicated user account (user:User@2026) and rejection of user:user
    user_auth = f"Basic {base64.b64encode(b'user:User@2026').decode()}"
    res_user = client.get("/api/v1/bot/time-limits", headers={"Authorization": user_auth})
    if res_user.status_code != 200:
        fail_check("User account access (user:User@2026)", str(res_user.status_code))
    pass_check("User account (user:User@2026) verified", "Status 200 OK")

    user_bad_auth = f"Basic {base64.b64encode(b'user:user').decode()}"
    res_bad_user = client.get("/api/v1/bot/time-limits", headers={"Authorization": user_bad_auth})
    if res_bad_user.status_code != 401:
        fail_check("Rejection of username as password (user:user)", f"Expected 401, got {res_bad_user.status_code}")
    pass_check("Username as password (user:user) properly rejected with 401", "Unauthorized")

    # 4d. Test pre-configured temporary trader account
    trader_auth = f"Basic {base64.b64encode(b'trader_1:Tk9#vL2pP').decode()}"
    res_trader = client.get("/api/v1/bot/time-limits", headers={"Authorization": trader_auth})
    if res_trader.status_code != 200:
        fail_check("Trader account access", str(res_trader.status_code))
    pass_check("Pre-configured trader account (trader_1) verified", "Status 200 OK")

    # 5. Test rejection of unapproved custom usernames with random passwords
    unapproved_auth = f"Basic {base64.b64encode(b'unregistered_user:random_pass_123').decode()}"
    res_unapproved = client.get("/api/v1/bot/time-limits", headers={"Authorization": unapproved_auth})
    if res_unapproved.status_code != 401:
        fail_check("Rejection of unregistered account", f"Expected 401, got {res_unapproved.status_code}")
    pass_check("Unregistered account properly rejected with HTTP 401", "Unauthorized")

    # 6. Test invalid credentials rejected with 401
    bad_auth = f"Basic {base64.b64encode(b'operator:wrong_password').decode()}"
    res_bad = client.get("/api/v1/bot/time-limits", headers={"Authorization": bad_auth})
    if res_bad.status_code != 401:
        fail_check("Rejection of bad password", f"Expected 401, got {res_bad.status_code}")
    pass_check("Bad credentials properly rejected with HTTP 401", "Unauthorized")

    # 7. Test expired/tampered trial token rejected
    tampered = "trial_1000000000_fakehash123456"
    bad_token_auth = f"Basic {base64.b64encode(f'guest:{tampered}'.encode()).decode()}"
    res_tampered = client.get("/api/v1/bot/time-limits", headers={"Authorization": bad_token_auth})
    if res_tampered.status_code != 401:
        fail_check("Rejection of forged trial token", f"Expected 401, got {res_tampered.status_code}")
    pass_check("Forged or expired trial token properly rejected", "Unauthorized")

    print("\n==================================================================")
    print("  ALL AUTH & TRIAL VERIFICATION CHECKS PASSED (100%)              ")
    print("==================================================================")


if __name__ == "__main__":
    main()
