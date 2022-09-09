# plot by number of dimensions
library(ggplot2)

convergence_bk_d50_slow = read.csv("convergence_bk_d50_slow.csv")
convergence_bps_d50_slow = read.csv("convergence_bps_d50_slow.csv")
convergence_hmc_d50_slow = read.csv("convergence_hmc_d50_slow.csv")

experiment = c(1, 2, 3, 4, 5)

df = data.frame(experiment, bk = convergence_bk_d50_slow$X0, bps = convergence_bps_d50_slow$X0, hmc = convergence_hmc_d50_slow$X0)

#plot 
png("convergence_slow.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=experiment)) + 
  geom_line(aes(y = hmc, color="NUTS-HMC")) +
  geom_line(aes(y = bk,  color = "BOOM")) + 
  xlab("Experiment") + 
  ggtitle("Few sparse models")+ 
  theme(plot.title = element_text(hjust = 0.5), legend.position="bottom") + 
  ylab("KSD convergence")+
  scale_colour_manual("", 
                      values = c("BOOM"="green", 
                                 "NUTS-HMC"="darkred")) 
dev.off()
