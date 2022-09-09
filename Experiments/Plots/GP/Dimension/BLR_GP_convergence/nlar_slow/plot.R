# plot by number of dimensions
library(ggplot2)

convergence_bk_d10_n1000_slow = read.csv("convergence_bk_d10_n1000_slow.csv")
convergence_zz_d10_n1000_slow = read.csv("convergence_zz_d10_n1000_slow.csv")
convergence_hmc_d10_n1000_slow = read.csv("convergence_hmc_d10_n1000_slow.csv")
convergence_bps_d10_n1000_slow = read.csv("convergence_bps_d10_n1000_slow.csv")
convergence_bk_d50_n1000_slow = read.csv("convergence_bk_d50_n1000_slow.csv")
convergence_zz_d50_n1000_slow = read.csv("convergence_zz_d50_n1000_slow.csv")
convergence_hmc_d50_n1000_slow = read.csv("convergence_hmc_d50_n1000_slow.csv")
convergence_bps_d50_n1000_slow = read.csv("convergence_bps_d50_n1000_slow.csv")
convergence_bk_d100_n10000_slow = read.csv("convergence_bk_d100_n10000_slow.csv")
convergence_zz_d100_n10000_slow = read.csv("convergence_zz_d100_n10000_slow.csv")
convergence_hmc_d100_n10000_slow = read.csv("convergence_hmc_d100_n10000_slow.csv")
convergence_bps_d100_n10000_slow = read.csv("convergence_bps_d100_n10000_slow.csv")

#dataframe creation 
convergence_d10 = c(convergence_bk_d10_n1000_slow$X0,convergence_bps_d10_n1000_slow$X0,convergence_zz_d10_n1000_slow$X0,convergence_hmc_d10_n1000_slow$X0)
convergence_d50 = c(convergence_bk_d50_n1000_slow$X0,convergence_bps_d50_n1000_slow$X0,convergence_zz_d50_n1000_slow$X0,convergence_hmc_d50_n1000_slow$X0)
convergence_d100 = c(convergence_bk_d100_n10000_slow$X0,convergence_bps_d100_n10000_slow$X0,convergence_zz_d100_n10000_slow$X0,convergence_hmc_d100_n10000_slow$X0)
convergence_val = log(c(convergence_d10, convergence_d50, convergence_d100))
convergence = c("10", "50", "100")
d_val = rep(convergence, each=40)
samplers = c("Boom", "BPS", "ZZ", "NUTS-HMC")
s_val = c(rep(samplers, each=10),rep(samplers, each=10),rep(samplers, each=10))

df = data.frame(d_val, convergence_val, s_val)

#plot 
png("convergence_by_dimension_nlar_slow.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=factor(d_val, level=c("10", "50", "100")), y=convergence_val, fill=s_val)) + 
  geom_boxplot() + xlab("Dimension") + ylab("log-scale KSD") + ggtitle("Low noise - Large size")+ labs(fill='Sampler') + theme(plot.title = element_text(hjust = 0.5), legend.position="bottom")
dev.off()