import express, { type Express } from "express";
import cors from "cors";
import pinoHttp from "pino-http";
import router from "./routes";
import healthRouter from "./routes/health";
import { rateLimit } from "./middlewares/rateLimit";
import { logger } from "./lib/logger";

const app: Express = express();

// Trust the Replit deployment proxy so req.ip / x-forwarded-for resolve correctly
app.set("trust proxy", true);

app.use(
  pinoHttp({
    logger,
    serializers: {
      req(req) {
        return { id: req.id, method: req.method, url: req.url?.split("?")[0] };
      },
      res(res) {
        return { statusCode: res.statusCode };
      },
    },
  }),
);
app.use(cors());
// Generous body limit — long Claude Code conversations can ship 1m-token context
app.use(express.json({ limit: "50mb" }));
app.use(express.urlencoded({ extended: true, limit: "50mb" }));

// Health route is mounted FIRST so it bypasses rate limiting (deployment probes
// hit it constantly and shouldn't count against the quota).
app.use("/", healthRouter);

// All proxy traffic is rate-limited per IP
app.use(rateLimit);

// Single mount point — clients use root-relative paths (/v1/messages, /v1/chat/completions)
app.use("/", router);

export default app;
