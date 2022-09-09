# plot by number of dimensions
library(ggplot2)

perCorrect_bk_d50_slow = read.csv("perCorrect_bk_d50_slow.csv")
perCorrect_bps_d50_slow = read.csv("perCorrect_bps_d50_slow.csv")
perCorrect_hmc_d50_slow = read.csv("perCorrect_hmc_d50_slow.csv")

experiment = c(1, 2, 3, 4, 5)

df = data.frame(experiment, bk = perCorrect_bk_d50_slow$X0, bps = perCorrect_bps_d50_slow$X0, hmc = perCorrect_hmc_d50_slow$X0)

#plot 
png("perCorrect_slow.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=experiment)) + 
  geom_line(aes(y = bps, color="BPS")) + 
  geom_line(aes(y = hmc, color="NUTS-HMC")) +
  geom_line(aes(y = bk,  color = "BOOM")) + 
  xlab("Experiment") + 
  ggtitle("Few sparse models")+ 
  theme(plot.title = element_text(hjust = 0.5), legend.position="bottom") + 
  ylab("Percentage of correctly detected values")+
  scale_colour_manual("", 
                      values = c("BOOM"="green", "BPS"="steelblue", 
                                 "NUTS-HMC"="darkred")) +
  ylim(0,1)
dev.off()
