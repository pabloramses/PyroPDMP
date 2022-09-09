# plot by number of dimensions
library(ggplot2)

r2scores_bk_d50_smed = read.csv("r2scores_bk_d50_smed.csv")
r2scores_bps_d50_smed = read.csv("r2scores_bps_d50_smed.csv")
r2scores_hmc_d50_smed = read.csv("r2scores_hmc_d50_smed.csv")

experiment = c(1, 2, 3, 4, 5)

df = data.frame(experiment, bk = r2scores_bk_d50_smed$X0, bps = r2scores_bps_d50_smed$X0, hmc = r2scores_hmc_d50_smed$X0)

#plot 
png("r2scores_smed.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=experiment)) + 
  geom_line(aes(y = bps, color="BPS")) + 
  geom_line(aes(y = hmc, color="NUTS-HMC")) +
  geom_line(aes(y = bk,  color = "BOOM")) + 
  xlab("Experiment") + 
  ggtitle("Mild sparse models")+ 
  theme(plot.title = element_text(hjust = 0.5), legend.position="bottom") + 
  ylab("R2 Scores")+
  scale_colour_manual("", 
                      values = c("BOOM"="green", "BPS"="steelblue", 
                                 "NUTS-HMC"="darkred")) 
dev.off()
