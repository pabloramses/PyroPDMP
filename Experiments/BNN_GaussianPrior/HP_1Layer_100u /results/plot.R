# plot by number of dimensions
library(ggplot2)

y_test = read.csv("y_test.csv")$X0
x_test = read.csv("x_test.csv")$X0
y_test_pred = read.csv("y_test_pred.csv")$X0

y_train = read.csv("y_train.csv")$X0
x_train = read.csv("x_train.csv")$X0
y_train_pred = read.csv("y_train_pred.csv")$X0

lower_test = read.csv("lower_test.csv")$X0
upper_test = read.csv("upper_test.csv")$X0

var_lower_test = read.csv("var_test_low.csv")$X0
var_upper_test = read.csv("var_test_up.csv")$X0


df = data.frame(x_test, y_test, y_test_pred, lower_test, upper_test, var_lower_test,var_upper_test, x_train, y_train)

y_ext = read.csv("y_ext.csv")$X0
x_ext = read.csv("x_ext.csv")$X0
y_ext_pred = read.csv("y_ext_pred.csv")$X0
lower_ext = read.csv("lower_ext.csv")$X0
upper_ext = read.csv("upper_ext.csv")$X0


#plot 
png("figure_BNN.png", units="in", width=7, height=5, res=600)
ggplot(df, aes(x_test)) +
  geom_ribbon(aes(ymin = lower_test, ymax = upper_test, color = "95% predictive CI"), fill = "grey70", linetype=0) + 
  geom_line(aes(y = y_test_pred, color = "Predictive MAP of test points")) + 
  geom_line(aes(y = y_test, color = "True values"), linetype="dashed") + 
  geom_point(aes(x = x_train, y = y_train, color="Train points")) +
  xlab("x") + 
  ylab("") +
  #scale_linetype_manual(values = c("Predictive MAP of test points" = "dashed","True values" = "dashed")) +
  scale_color_manual(name="", values = c("Train points"="grey40", "Predictive MAP of test points" = "red","True values" = "black", "95% predictive CI"="grey"))+
  guides(color = guide_legend(override.aes = list(linetype = c(0, 1, 2, 0),
                                                  shape = c(16,NA,NA,NA), 
                                                  size = 1)))+

  theme(legend.position = c(0.2, 0.85),
        legend.background = element_rect(fill = "white"),
        legend.title = element_text(size=3),
        legend.text = element_text(size=5),
        legend.key = element_rect(size=10)) 

                       
                       
#c("95% predictive CI" = "grey","Train points" = "black")) #+
  #guides(color = guide_legend(override.aes = list(linetype = c(1, "dashed", "dashed"),
                                                  #shape = c(NA))))+theme(plot.title = element_text(hjust = 0.5, size = 10))

dev.off()

png("figure_BNN_extended.png", units="in", width=7, height=5, res=600)
df_complete = data.frame(x = c(x_test, x_ext),  y_true = c(y_test, y_ext), y_pred = c(y_test_pred, y_ext_pred),  low = c(lower_test, lower_ext), up = c(upper_test, upper_ext), x_train, y_train)
ggplot(df_complete, aes(x)) +
  geom_ribbon(aes(ymin = low, ymax = up, color = "95% predictive CI"), fill = "grey70", linetype=0) + 
  geom_line(aes(y = y_true), linetype="dashed") + 
  geom_line(aes(y = y_pred)) + 
  geom_point(aes(x = x_train, y = y_train)) +
  xlab("x") + 
  ylab("") + 
  theme(legend.position = c(0.17, 0.9),
        legend.background = element_rect(fill = "white", color = "black"),
        legend.key.size = unit(0.1, 'cm'))
dev.off()
