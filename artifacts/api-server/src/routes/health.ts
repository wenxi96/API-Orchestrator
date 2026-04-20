import { Router, type IRouter } from "express";
import { authReady } from "../lib/authWarmup.js";

const router: IRouter = Router();

router.get("/healthz", (_req, res) => {
  // Return 503 until at least one model is ready, so deployment platforms
  // don't route traffic to a server that will reject every request.
  if (!authReady) {
    res.status(503).json({ status: "starting", authReady: false });
    return;
  }
  res.json({ status: "ok", authReady: true });
});

export default router;
