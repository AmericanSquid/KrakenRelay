import threading
import logging

class Lifecycle:
    def __init__(self, controller):
        self.controller = controller

        self._shutdown_failed = False
        self._cleanup_lock = threading.Lock()

    def start(self):
        controller = self.controller
        tx = controller.tx_state

        if controller.running:
            logging.warning("Repeater is already running.")
            return

        controller.running = True
        self.audio_thread = threading.Thread(target=controller.audio_loop.audio_loop)
        self.audio_thread.start()
        logging.info("Repeater controller started")

        logging.info(f"🧵 Active threads after start: {threading.active_count()}")
        for t in threading.enumerate():
            try:
                ident = t.ident  # Python thread ident
                # On Linux, get native TID (will match top/htop)
                tid = t.native_id if hasattr(t, 'native_id') else None
                logging.info(f"Thread name: {t.name}, ident: {ident}, native_id: {tid}")
            except Exception as e:
                logging.info(f"Could not get info for thread: {t} ({e})") 

    def cleanup(self) -> bool:
        controller = self.controller
        tx = controller.tx_state

        logging.info("========== [Repeater] Cleanup Starting ==========")
        logging.info(
            "Controller state before cleanup: running=%s transmitting=%s skip_courtesy_tone=%s shutdown_failed=%s",
            controller.running, tx.transmitting, tx.skip_courtesy_tone, self._shutdown_failed
        )
        ok = True
        self._shutdown_failed = False

        # --- A) SAFETY FIRST: UNKEY ---
        logging.info("[Cleanup Step A] Unkeying transmitter and suppressing courtesy tone...")
        try:
            tx.transmitting = False
            tx.skip_courtesy_tone = True
            controller.ptt_manager.safe_ptt_unkey()
            logging.info("[Cleanup Step A] PTT unkeyed (forced cleanup).")
        except Exception:
            logging.exception("[Cleanup Step A] Failed to unkey PTT during cleanup.")
            ok = False

        # --- B) SIGNAL ALL LOOPS/THREADS TO STOP ---
        controller.running = False  # Stop the loop
        logging.info("[Cleanup Step B] Stopping audio loop: running=%s", controller.running)

        # --- C) JOIN THREADS ---
        def _join(th, name: str, timeout: float) -> bool:
            if not th:
                return True
            if not th.is_alive():
                return True
            if th is threading.current_thread():
                logging.warning("[Repeater] cleanup called from %s; skipping self-join", name)
                return True
            logging.info("[Cleanup Step C] Joining thread: %s", name)
            th.join(timeout=timeout)
            if th.is_alive():
                logging.error("[Repeater] %s still alive after %.1fs", name, timeout)
                return False
            logging.info("[Cleanup Step C] Joined thread: %s", name)
            return True

        ok &= _join(getattr(self, "audio_thread", None), "audio_thread", 3.0)

        if not ok:
            logging.critical("[Repeater] cleanup incomplete - leaving PortAudio intact. Restarting service to recover.")
            self._shutdown_failed = True
            return False

        # --- D) CLOSE STREAMS (threads are gone) ---
        if ok:
            for name in ("input_stream", "output_stream", "output_stream_2"):
                s = getattr(controller.streams, name, None)
                if not s:
                    continue
                try:
                    logging.info(f"[Cleanup Step D] Closing stream: {name}")
                    try:
                        s.stop_stream()
                    except Exception:
                        pass
                    s.close()
                    logging.info(f"[Cleanup Step D] Closed stream: {name}")
                except Exception as e:
                    logging.warning(f"[Cleanup Step D] Exception closing {name}: {e}")
                    ok = False
                setattr(controller, name, None)

        self._shutdown_failed = not ok
        if ok:
            logging.info("Cleanup completed successfully. No issues detected.")
        else:
            logging.warning("Cleanup completed with ERRORS. Check previous logs for details.")

        logging.info("[Repeater] cleanup() done. ok=%s shutdown_failed=%s", ok, self._shutdown_failed)
        logging.info("🧵 Active threads after cleanup: %d", threading.active_count())
        for t in threading.enumerate():
            logging.info("  • %s (alive=%s)", t.name, t.is_alive())

        logging.info("========== [Repeater] Clean Up Ended ==========")
        logging.info("73 de K3AYV (American Squid).")
        return ok
