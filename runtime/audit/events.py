# runtime/audit/events.py

from enum import Enum


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditEvent(str, Enum):
    # ============================================================
    # Lifecycle
    # ============================================================

    SYSTEM_START = "SYSTEM_START"
    SYSTEM_STOP = "SYSTEM_STOP"

    CONTROLLER_START = "CONTROLLER_START"
    CONTROLLER_STOP = "CONTROLLER_STOP"
    CONTROLLER_CRASH = "CONTROLLER_CRASH"

    AUTO_RESTART = "AUTO_RESTART"

    # ============================================================
    # Authentication / Attribution
    # ============================================================

    USER_LOGIN = "USER_LOGIN"
    USER_LOGOUT = "USER_LOGOUT"

    AUTH_FAILURE = "AUTH_FAILURE"
    PERMISSION_DENIED = "PERMISSION_DENIED"

    # ============================================================
    # Configuration
    # ============================================================

    CONFIG_CHANGED = "CONFIG_CHANGED"
    CONFIG_APPLIED = "CONFIG_APPLIED"
    CONFIG_RELOADED = "CONFIG_RELOADED"

    CONFIG_SAVE_FAILED = "CONFIG_SAVE_FAILED"
    CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"

    # ============================================================
    # Admin / Service Actions
    # ============================================================

    SERVICE_RESTART_REQUESTED = "SERVICE_RESTART_REQUESTED"
    SERVICE_RESTARTED = "SERVICE_RESTARTED"

    SYSTEM_REBOOT_REQUESTED = "SYSTEM_REBOOT_REQUESTED"

    MANUAL_RECOVERY_TRIGGERED = "MANUAL_RECOVERY_TRIGGERED"

    # ============================================================
    # Hardware / Audio Devices
    # ============================================================

    AUDIO_DEVICE_LOST = "AUDIO_DEVICE_LOST"
    AUDIO_DEVICE_RECOVERED = "AUDIO_DEVICE_RECOVERED"

    AUDIO_DEVICE_DISCONNECTED = "AUDIO_DEVICE_DISCONNECTED"
    AUDIO_DEVICE_RECONNECTED = "AUDIO_DEVICE_RECONNECTED"

    AUDIO_STREAM_TIMEOUT = "AUDIO_STREAM_TIMEOUT"

    AUDIO_STREAM_READ_FAILURE = "AUDIO_STREAM_READ_FAILURE"
    AUDIO_STREAM_WRITE_FAILURE = "AUDIO_STREAM_WRITE_FAILURE"

    USB_STREAM_STALLED = "USB_STREAM_STALLED"

    STREAM_REOPENED = "STREAM_REOPENED"

    PYAUDIO_OVERFLOW = "PYAUDIO_OVERFLOW"
    PYAUDIO_UNDERFLOW = "PYAUDIO_UNDERFLOW"

    # ============================================================
    # Watchdog / Recovery
    # ============================================================

    WATCHDOG_TRIGGERED = "WATCHDOG_TRIGGERED"

    WATCHDOG_RECOVERY_STARTED = "WATCHDOG_RECOVERY_STARTED"
    WATCHDOG_RECOVERY_SUCCEEDED = "WATCHDOG_RECOVERY_SUCCEEDED"
    WATCHDOG_RECOVERY_FAILED = "WATCHDOG_RECOVERY_FAILED"

    WATCHDOG_FLAPPING = "WATCHDOG_FLAPPING"

    # ============================================================
    # PTT
    # ============================================================

    PTT_DEVICE_MISSING = "PTT_DEVICE_MISSING"
    PTT_FAILURE = "PTT_FAILURE"

    PTT_STUCK_ACTIVE = "PTT_STUCK_ACTIVE"

    # ============================================================
    # Radio / RF / Repeater Anomalies
    # ============================================================

    TOT_TRIGGERED = "TOT_TRIGGERED"

    INVALID_CARRIER = "INVALID_CARRIER"
    CARRIER_VALIDITY_FAILED = "CARRIER_VALIDITY_FAILED"

    STUCK_TX_DETECTED = "STUCK_TX_DETECTED"

    NO_AUDIO_DURING_TX = "NO_AUDIO_DURING_TX"

    RAPID_KEYING_DETECTED = "RAPID_KEYING_DETECTED"

    LONG_DURATION_TX = "LONG_DURATION_TX"

    EXCESSIVE_TX_DUTY_CYCLE = "EXCESSIVE_TX_DUTY_CYCLE"

    # ============================================================
    # Audit System
    # ============================================================

    AUDIT_WRITE_FAILURE = "AUDIT_WRITE_FAILURE"
    AUDIT_DB_CORRUPTION = "AUDIT_DB_CORRUPTION"
    AUDIT_QUEUE_OVERFLOW = "AUDIT_QUEUE_OVERFLOW"

    # ============================================================
    # Maintenance
    # ============================================================

    MAINTENANCE_MODE_ENABLED = "MAINTENANCE_MODE_ENABLED"
    MAINTENANCE_MODE_DISABLED = "MAINTENANCE_MODE_DISABLED"
