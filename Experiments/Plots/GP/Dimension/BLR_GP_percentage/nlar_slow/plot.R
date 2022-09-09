# plot by number of dimensions
library(ggplot2)

perCorrect_bk_d10_n1000_slow = read.csv("perCorrect_bk_d10_n1000_slow.csv")
perCorrect_zz_d10_n1000_slow = read.csv("perCorrect_zz_d10_n1000_slow.csv")
perCorrect_hmc_d10_n1000_slow = read.csv("perCorrect_hmc_d10_n1000_slow.csv")
perCorrect_bps_d10_n1000_slow = read.csv("perCorrect_bps_d10_n1000_slow.csv")
perCorrect_bk_d50_n1000_slow = read.csv("perCorrect_bk_d50_n1000_slow.csv")
perCorrect_zz_d50_n1000_slow = read.csv("perCorrect_zz_d50_n1000_slow.csv")
perCorrect_hmc_d50_n1000_slow = read.csv("perCorrect_hmc_d50_n1000_slow.csv")
perCorrect_bps_d50_n1000_slow = read.csv("perCorrect_bps_d50_n1000_slow.csv")
perCorrect_bk_d100_n10000_slow = read.csv("perCorrect_bk_d100_n10000_slow.csv")
perCorrect_zz_d100_n10000_slow = read.csv("perCorrect_zz_d100_n10000_slow.csv")
perCorrect_hmc_d100_n10000_slow = read.csv("perCorrect_hmc_d100_n10000_slow.csv")
perCorrect_bps_d100_n10000_slow = read.csv("perCorrect_bps_d100_n10000_slow.csv")

#dataframe creation 
perCorrect_d10 = c(perCorrect_bk_d10_n1000_slow$X0,perCorrect_bps_d10_n1000_slow$X0,perCorrect_zz_d10_n1000_slow$X0,perCorrect_hmc_d10_n1000_slow$X0)
perCorrect_d50 = c(perCorrect_bk_d50_n1000_slow$X0,perCorrect_bps_d50_n1000_slow$X0,perCorrect_zz_d50_n1000_slow$X0,perCorrect_hmc_d50_n1000_slow$X0)
perCorrect_d100 = c(perCorrect_bk_d100_n10000_slow$X0,perCorrect_bps_d100_n10000_slow$X0,perCorrect_zz_d100_n10000_slow$X0,perCorrect_hmc_d100_n10000_slow$X0)
perCorrect_val = c(perCorrect_d10, perCorrect_d50, perCorrect_d100)
perCorrect = c("10", "50", "100")
d_val = rep(perCorrect, each=40)
samplers = c("Boom", "BPS", "ZZ", "NUTS-HMC")
s_val = c(rep(samplers, each=10),rep(samplers, each=10),rep(samplers, each=10))

df = data.frame(d_val, perCorrect_val, s_val)

#plot 
png("perCorrect_by_dimension_nlar_slow.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=factor(d_val, level=c("10", "50", "100")), y=perCorrect_val, fill=s_val)) + 
  geom_boxplot() + xlab("Dimension") + ylab("Percentage correct values detected") + ggtitle("Low noise - Large size")+ labs(fill='Sampler') + theme(plot.title = element_text(hjust = 0.5), legend.position="bottom")
dev.off()