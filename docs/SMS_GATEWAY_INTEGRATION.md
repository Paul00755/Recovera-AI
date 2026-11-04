# OnePlus 10T 5G (Android 15) SMS Gateway Integration

## 1. Context & Device Specification

- **Primary Device**: OnePlus 10T 5G
- **Operating System**: Android 15 (API Level 35)
- **Sender Hardware**: Device Primary SIM (carrier cellular network)
- **Function**: Dedicated local SMS transmission gateway for post-discharge patient care notifications.
- **Direction**: Strict **One-Way Outbound**. No patient replies are accepted or processed via SMS.

## 2. Technical Challenge & Solution Topology

Desktop Python running on Windows cannot directly access a smartphone's hardware baseband/SIM module without an intermediary bridge. Furthermore, Android 15 enforces strict background execution limits, battery optimization rules, and runtime permission scopes (`SEND_SMS`, `READ_PHONE_STATE`).

### Selected Topology: Secured HTTP Companion Bridge

```
+------------------------------------+             +-----------------------------------------+
|     Recovera AI Backend (FastAPI)   |             |       OnePlus 10T 5G (Android 15)        |
|                                    |             |                                         |
|  +------------------------------+  |             |  +-----------------------------------+  |
|  | SmsGatewayService            |  |  HTTP POST  |  | SMS Gateway Companion App         |  |
|  |  - Outbound Queue            |  |  (Bearer    |  |  - Foreground Service (Wakelock)  |  |
|  |  - Rate Limiter (1 msg/3s)   |==================>|  - Port 8080 (Local Wi-Fi/USB)     |  |
|  |  - Bounded Retries (max 3)   |  Auth Token)|  |  - Android SmsManager API            |  |
|  |  - Gateway Health Heartbeat  |  |             |  +-----------------+-----------------+  |
|  +------------------------------+  |             |                    |                    |
+------------------------------------+             |                    v                    |
                                                   |     [Primary SIM Cellular Network]       |
                                                   +--------------------+--------------------+
                                                                        |
                                                                        v
                                                   +-----------------------------------------+
                                                   |         Patient Mobile Device           |
                                                   |  "Medication reminder: please follow..." |
                                                   +-----------------------------------------+
```

### Protocol & API Contract

1. **Send SMS**:
   - `POST /api/v1/sms/send`
   - Headers: `Authorization: Bearer <CONFIGURED_GATEWAY_TOKEN>`
   - Request Body:
     ```json
     {
       "to": "+919876543210",
       "message": "Medication reminder: please follow your approved prescription — Amoxicillin 500mg, After food. Open Recovera for your schedule.",
       "idempotency_key": "occ-550e8400-e29b-41d4-a716-446655440000-dose"
     }
     ```
   - Response:
     ```json
     {
       "status": "accepted",
       "message_id": "sms-oneplus-987162",
       "queued_at": "2026-10-12T19:40:00Z"
     }
     ```

2. **Device Health Check**:
   - `GET /api/v1/health`
   - Headers: `Authorization: Bearer <CONFIGURED_GATEWAY_TOKEN>`
   - Response:
     ```json
     {
       "status": "online",
       "battery_pct": 87,
       "charging": true,
       "sim_operator": "Airtel 5G",
       "sim_slot": 1,
       "signal_strength_dbm": -85,
       "pending_queue_count": 0
     }
     ```

## 3. Android 15 Permissions & System Settings

To prevent Android 15 from killing the gateway process when the phone screen is off:
1. **Runtime Permissions Granted**:
   - `android.permission.SEND_SMS`
   - `android.permission.READ_PHONE_STATE`
   - `android.permission.POST_NOTIFICATIONS`
2. **Battery Optimization**: Set app battery usage to **Unrestricted** in Settings $\rightarrow$ Apps $\rightarrow$ Battery.
3. **Foreground Notification**: The companion runs an active foreground service displaying persistent notification *"Recovera SMS Gateway Running"*.
4. **Network Access**: Connected either via local hospital Wi-Fi with static local IP (e.g. `http://192.168.1.150:8080`) or via USB reverse tethering (`adb forward tcp:8080 tcp:8080` $\rightarrow$ `http://localhost:8080`).

## 4. Bounded Retries, Rate Limiting & Offline Protection

- **Rate Limiting**: Dispatches are metered to maximum 1 SMS every 3 seconds to avoid carrier throttling or SIM spam flagging.
- **Daily Volume Cap**: Enforces a configurable daily volume cap (default 500 SMS/day) to prevent accidental runaway billing.
- **Bounded Retry**: If the device returns a non-200 HTTP code or timeout occurs, the message enters a retry backoff (intervals: 1m, 3m, 5m), failing permanently after 3 attempts.
- **Offline Notification & Fallback**:
  - The Admin and Nurse dashboards display a live visual indicator (`Connected` / `Disconnected`).
  - If the device is disconnected, outbound jobs remain safely in `QUEUED` state until connection is restored.
  - A mock test driver (`MockDeviceDriver`) is provided so automated testing and offline development function cleanly without blocking on physical hardware availability.
