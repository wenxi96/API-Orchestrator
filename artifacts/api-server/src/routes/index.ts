import { Router, type IRouter } from "express";
import openaiCompatRouter from "./proxy/openaiCompat.js";
import anthropicCompatRouter from "./proxy/anthropicCompat.js";

const router: IRouter = Router();

// Health is mounted separately in app.ts (before the rate limiter)
router.use("/v1", openaiCompatRouter);
router.use("/v1", anthropicCompatRouter);

export default router;
