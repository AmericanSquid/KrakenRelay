import logging
import socket


class UVK5DPTT:
    """
    PTT backend for uvk5d.

    Flow:
        KrakenRelay -> uvk5d TCP socket -> UV-K5 serial -> radio
    """

    def __init__(self, host="127.0.0.1", port=7355, timeout=0.5):
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)

        # PTTManager status/logging code expects these attributes.
        self.device = f"{self.host}:{self.port}"
        self.pin = None
        self.working = True

        logging.info("[UVK5DPTT] Initialized uvk5d backend at %s", self.device)

    def _send(self, command):
        command = str(command).strip()

        try:
            with socket.create_connection(
                (self.host, self.port),
                timeout=self.timeout,
            ) as sock:
                sock.settimeout(self.timeout)
                sock.sendall((command + "\n").encode("ascii"))

                chunks = []

                while True:
                    try:
                        data = sock.recv(4096)
                    except socket.timeout:
                        break

                    if not data:
                        break

                    chunks.append(data)
                    text = b"".join(chunks).decode("utf-8", errors="replace")

                    if text.startswith("OK ") or "\nOK " in text:
                        break

                    if text.startswith("ERR ") or "\nERR " in text:
                        break

                    if "END DAEMON" in text or "END STAT" in text or "END TXP" in text:
                        break

                response = b"".join(chunks).decode("utf-8", errors="replace").strip()

        except Exception as exc:
            self.working = False
            logging.error("[UVK5DPTT] Command failed: %s: %s", command, exc)
            raise

        if response.startswith("ERR "):
            self.working = False
            logging.error("[UVK5DPTT] uvk5d error for %s: %s", command, response)
            raise RuntimeError(response)

        self.working = True
        return response

    def key(self):
        self._send("PTT ON")

    def unkey(self):
        self._send("PTT OFF")

    def status(self):
        try:
            response = self._send("DAEMON?")
        except Exception:
            return (f"UVK5D Offline ({self.device})", "red")

        if "STATUS=OK" in response or response.startswith("BEGIN DAEMON"):
            return (f"UVK5D OK ({self.device})", "green")

        return (f"UVK5D Weird Response ({self.device})", "orange")
