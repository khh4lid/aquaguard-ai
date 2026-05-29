"""
Alert delivery: Twilio WhatsApp + GPIO siren.

Notes that matter on real hardware:
- gpiozero (with the lgpio backend) is the correct GPIO stack on Pi 5.
  RPi.GPIO does NOT work on the Pi 5's new GPIO chip — do not use it.
- WhatsApp sends are fired on a daemon thread so a slow network call never
  stalls the video loop (a frozen loop = missed alerts).
- Photo evidence is saved locally. Twilio WhatsApp media needs a PUBLICLY
  reachable URL, so on-device we send text by default; wire media_url later
  if you host the evidence image somewhere public.
"""
import threading
import time
from datetime import datetime
from . import config


class AlertManager:
    def __init__(self):
        self._twilio = None
        self._siren = None
        self._init_twilio()
        self._init_siren()

    def _init_twilio(self):
        if config.TWILIO_SID and config.TWILIO_TOKEN and config.PARENT_WHATSAPP:
            try:
                from twilio.rest import Client
                self._twilio = Client(config.TWILIO_SID, config.TWILIO_TOKEN)
                print("[alerts] Twilio ready")
            except Exception as e:
                print(f"[alerts] Twilio init failed: {e}")
        else:
            print("[alerts] Twilio not configured — alerts print to console only")

    def _init_siren(self):
        try:
            from gpiozero import OutputDevice  # drives a relay or active buzzer
            self._siren = OutputDevice(config.SIREN_PIN, active_high=True,
                                       initial_value=False)
            print(f"[alerts] Siren ready on GPIO{config.SIREN_PIN}")
        except Exception as e:
            print(f"[alerts] Siren unavailable ({e}) — running without GPIO")

    def _send_whatsapp(self, body):
        ts = datetime.now().strftime("%H:%M:%S")
        if self._twilio:
            try:
                self._twilio.messages.create(
                    body=body, from_=config.TWILIO_FROM, to=config.PARENT_WHATSAPP)
                print(f"[alerts] WhatsApp sent @ {ts}")
            except Exception as e:
                print(f"[alerts] WhatsApp failed: {e}")
        else:
            print(f"\n[ALERT @ {ts}]\n{body}\n")

    def _wail_siren(self):
        if not self._siren:
            return
        self._siren.on()
        time.sleep(config.SIREN_SECONDS)
        self._siren.off()

    def danger(self, evidence_path=None):
        body = ("DANGER — AquaGuard AI\n"
                f"{datetime.now():%H:%M:%S}\n"
                "Child near pool, no adult visible. Please check now.")
        threading.Thread(target=self._send_whatsapp, args=(body,), daemon=True).start()

    def emergency(self, evidence_path=None):
        body = ("EMERGENCY — AquaGuard AI\n"
                f"{datetime.now():%H:%M:%S}\n"
                "CHILD INSIDE POOL, NO ADULT. Immediate response required.")
        threading.Thread(target=self._send_whatsapp, args=(body,), daemon=True).start()
        threading.Thread(target=self._wail_siren, daemon=True).start()

    def cleanup(self):
        if self._siren:
            self._siren.off()
            self._siren.close()

    def heartbeat(self, body):
        """Low-priority 'system alive' ping (no siren). Sent on a daemon thread."""
        threading.Thread(target=self._send_whatsapp, args=(body,), daemon=True).start()
