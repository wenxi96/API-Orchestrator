import { Router, type IRouter } from "express";
import { authReady } from "../lib/authWarmup.js";

const router: IRouter = Router();

router.get("/healthz", (_req, res) => {
  res.json({ status: "ok", authReady });
});

export default router;
