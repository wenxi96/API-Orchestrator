import { Router, type IRouter } from "express";
import healthRouter from "./health.js";
import openaiCompatRouter from "./proxy/openaiCompat.js";
import anthropicCompatRouter from "./proxy/anthropicCompat.js";

const router: IRouter = Router();

router.use(healthRouter);
router.use("/v1", openaiCompatRouter);
router.use("/v1", anthropicCompatRouter);

export default router;
