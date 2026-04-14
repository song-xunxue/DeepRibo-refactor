#'
#' DeepRibo 训练日志可视化
#'
#' 作者: 李文煜
#' 日期: 2026-04-14
#'
#' 读取训练过程中产生的 training_log.csv，绘制：
#'   1. 训练集 vs 验证集 Loss 对比图
#'   2. 训练集 vs 验证集 Accuracy 对比图
#'
#' 用法:
#'   Rscript src/analysis/plot_training.R <training_log.csv> [输出目录]
#'
#' 示例:
#'   Rscript src/analysis/plot_training.R models/eco/2026-04-14-13-34/training_log.csv
#'   Rscript src/analysis/plot_training.R models/eco/2026-04-14-13-34/training_log.csv figures/eco
#'

library(ggplot2)

# ---- 参数解析 ----

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("用法: Rscript plot_training.R <training_log.csv> [输出目录]\n",
       "示例: Rscript plot_training.R models/eco/2026-04-14-13-34/training_log.csv figures/eco",
       call. = FALSE)
}

log_file  <- args[1]
out_dir   <- if (length(args) >= 2) args[2] else dirname(log_file)

if (!file.exists(log_file)) {
  stop("文件不存在: ", log_file, call. = FALSE)
}

# ---- 读取训练日志 ----

# 跳过前13行元数据（# 开头），从第14行（列头）开始读取
df <- read.csv(log_file, skip = 13, stringsAsFactors = FALSE)

cat(sprintf("已读取 %d 个 epoch 的训练数据\n", nrow(df)))
cat(sprintf("  训练集: loss=%.4f, acc=%.4f, auc=%.4f\n",
            tail(df$train_loss, 1), tail(df$train_acc, 1), tail(df$train_auc, 1)))
cat(sprintf("  验证集: loss=%.4f, acc=%.4f, auc=%.4f\n",
            tail(df$valid_loss, 1), tail(df$valid_acc, 1), tail(df$valid_auc, 1)))

# 读取元数据中的数据集名称
meta_lines <- readLines(log_file, n = 13)
dataset_name <- gsub("^# dataset=", "", grep("^# dataset=", meta_lines, value = TRUE))
if (length(dataset_name) == 0) dataset_name <- "unknown"

# ---- 确保输出目录存在 ----

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  cat(sprintf("创建输出目录: %s\n", out_dir))
}

# ---- 绘图主题 ----

theme_deepribo <- theme_minimal(base_size = 14) +
  theme(
    plot.title    = element_text(hjust = 0.5, face = "bold", size = 16),
    plot.subtitle = element_text(hjust = 0.5, color = "grey40"),
    legend.position  = "top",
    legend.title     = element_text(face = "bold"),
    panel.grid.minor = element_blank()
  )

# 配色
color_train <- "#E64B35"   # 红色系 - 训练集
color_valid <- "#4DBBD5"   # 蓝色系 - 验证集

# ---- 图1: Loss 对比图 ----

p_loss <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = train_loss, color = "训练集"), linewidth = 1.2) +
  geom_point(aes(y = train_loss, color = "训练集"), size = 2.5) +
  geom_line(aes(y = valid_loss, color = "验证集"), linewidth = 1.2) +
  geom_point(aes(y = valid_loss, color = "验证集"), size = 2.5) +
  scale_color_manual(
    name   = "数据集",
    values = c("训练集" = color_train, "验证集" = color_valid)
  ) +
  scale_x_continuous(breaks = seq(1, max(df$epoch), by = ceiling(max(df$epoch) / 10))) +
  labs(
    title    = paste0("Loss 曲线 (", dataset_name, ")"),
    subtitle = "训练集 vs 验证集",
    x        = "Epoch",
    y        = "Loss",
    color    = "数据集"
  ) +
  theme_deepribo

# ---- 图2: Accuracy 对比图 ----

p_acc <- ggplot(df, aes(x = epoch)) +
  geom_line(aes(y = train_acc, color = "训练集"), linewidth = 1.2) +
  geom_point(aes(y = train_acc, color = "训练集"), size = 2.5) +
  geom_line(aes(y = valid_acc, color = "验证集"), linewidth = 1.2) +
  geom_point(aes(y = valid_acc, color = "验证集"), size = 2.5) +
  scale_color_manual(
    name   = "数据集",
    values = c("训练集" = color_train, "验证集" = color_valid)
  ) +
  scale_x_continuous(breaks = seq(1, max(df$epoch), by = ceiling(max(df$epoch) / 10))) +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  labs(
    title    = paste0("Accuracy 曲线 (", dataset_name, ")"),
    subtitle = "训练集 vs 验证集",
    x        = "Epoch",
    y        = "Accuracy",
    color    = "数据集"
  ) +
  theme_deepribo

# ---- 保存图片 ----

loss_path <- file.path(out_dir, "loss_curve.png")
acc_path  <- file.path(out_dir, "acc_curve.png")

ggsave(loss_path, p_loss, width = 8, height = 5, dpi = 300)
ggsave(acc_path,  p_acc,  width = 8, height = 5, dpi = 300)

cat(sprintf("\n图片已保存:\n"))
cat(sprintf("  Loss:       %s\n", loss_path))
cat(sprintf("  Accuracy:   %s\n", acc_path))
