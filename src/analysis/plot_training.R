#'
#' DeepRibo 训练日志可视化
#'
#' 作者: 李文煜
#' 日期: 2026-04-14
#'
#' 2026-04-14
#' 变更说明：
#'   1. 删除图例标题"数据集"，图例仅显示"训练/验证"
#'   2. 美化：添加半透明面积填充、圆角背景、更精致的点线样式
#'
#' 读取训练过程中产生的 training_log.csv，绘制：
#'   1. 训练 vs 验证 Loss 对比图
#'   2. 训练 vs 验证 Accuracy 对比图
#'
#' 用法:
#'   Rscript src/analysis/plot_training.R <training_log.csv> [输出目录]
#'

library(ggplot2)
library(scales)

# ---- 参数解析 ----

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("用法: Rscript plot_training.R <training_log.csv> [输出目录]",
       call. = FALSE)
}

log_file  <- args[1]
out_dir   <- if (length(args) >= 2) args[2] else dirname(log_file)

if (!file.exists(log_file)) {
  stop("文件不存在: ", log_file, call. = FALSE)
}

# ---- 读取训练日志 ----

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

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
  cat(sprintf("创建输出目录: %s\n", out_dir))
}

# ---- 绘图主题 ----

theme_deepribo <- theme_minimal(base_size = 13) +
  theme(
    plot.title        = element_text(hjust = 0.5, face = "bold", size = 15,
                                     margin = margin(b = 6)),
    plot.subtitle     = element_text(hjust = 0.5, color = "grey50", size = 10),
    legend.position   = "top",
    legend.title      = element_blank(),
    legend.text       = element_text(size = 12),
    legend.key.width  = unit(2, "cm"),
    legend.key.height = unit(0.4, "cm"),
    axis.title        = element_text(size = 12, color = "grey30"),
    axis.text         = element_text(size = 11, color = "grey40"),
    panel.grid.major  = element_line(color = "grey90", linewidth = 0.4),
    panel.grid.minor  = element_blank(),
    plot.background   = element_rect(fill = "white", color = NA),
    plot.margin       = margin(10, 12, 10, 12)
  )

# 配色
color_train <- "#E64B35"
color_valid <- "#3C6CDF"
fill_train  <- alpha(color_train, 0.08)
fill_valid  <- alpha(color_valid, 0.08)

epoch_breaks <- seq(1, max(df$epoch), by = ceiling(max(df$epoch) / 10))

# ---- 图1: Loss 对比图 ----

p_loss <- ggplot(df, aes(x = epoch)) +
  geom_ribbon(aes(ymin = min(train_loss, valid_loss), ymax = train_loss),
              fill = fill_train) +
  geom_ribbon(aes(ymin = min(train_loss, valid_loss), ymax = valid_loss),
              fill = fill_valid) +
  geom_line(aes(y = train_loss, color = "Train"), linewidth = 1) +
  geom_point(aes(y = train_loss, color = "Train"), size = 2, shape = 16) +
  geom_line(aes(y = valid_loss, color = "Valid"), linewidth = 1) +
  geom_point(aes(y = valid_loss, color = "Valid"), size = 2, shape = 17) +
  scale_color_manual(
    values = c("Train" = color_train, "Valid" = color_valid)
  ) +
  scale_x_continuous(breaks = epoch_breaks) +
  labs(
    title = paste0("Loss Curve (", toupper(dataset_name), ")"),
    x     = "Epoch",
    y     = "Loss"
  ) +
  theme_deepribo

# ---- 图2: Accuracy 对比图 ----

p_acc <- ggplot(df, aes(x = epoch)) +
  geom_ribbon(aes(ymin = min(train_acc, valid_acc), ymax = train_acc),
              fill = fill_train) +
  geom_ribbon(aes(ymin = min(train_acc, valid_acc), ymax = valid_acc),
              fill = fill_valid) +
  geom_line(aes(y = train_acc, color = "Train"), linewidth = 1) +
  geom_point(aes(y = train_acc, color = "Train"), size = 2, shape = 16) +
  geom_line(aes(y = valid_acc, color = "Valid"), linewidth = 1) +
  geom_point(aes(y = valid_acc, color = "Valid"), size = 2, shape = 17) +
  scale_color_manual(
    values = c("Train" = color_train, "Valid" = color_valid)
  ) +
  scale_x_continuous(breaks = epoch_breaks) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = paste0("Accuracy Curve (", toupper(dataset_name), ")"),
    x     = "Epoch",
    y     = "Accuracy"
  ) +
  theme_deepribo

# ---- 保存图片 ----

loss_path <- file.path(out_dir, "loss_curve.png")
acc_path  <- file.path(out_dir, "acc_curve.png")

ggsave(loss_path, p_loss, width = 8, height = 5, dpi = 300, bg = "white")
ggsave(acc_path,  p_acc,  width = 8, height = 5, dpi = 300, bg = "white")

cat(sprintf("\n图片已保存:\n"))
cat(sprintf("  Loss:       %s\n", loss_path))
cat(sprintf("  Accuracy:   %s\n", acc_path))
