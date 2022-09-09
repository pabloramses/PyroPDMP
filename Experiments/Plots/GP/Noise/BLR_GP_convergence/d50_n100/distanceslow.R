# plot by number of dimensions
library(ggplot2)

distances_bk_d10_n100_slow = read.csv("distances_bk_d10_n100_slow.csv")
distances_zz_d10_n100_slow = read.csv("distances_zz_d10_n100_slow.csv")
distances_hmc_d10_n100_slow = read.csv("distances_hmc_d10_n100_slow.csv")
distances_bps_d10_n100_slow = read.csv("distances_bps_d10_n100_slow.csv")
distances_bk_d50_n100_slow = read.csv("distances_bk_d50_n100_slow.csv")
distances_zz_d50_n100_slow = read.csv("distances_zz_d50_n100_slow.csv")
distances_hmc_d50_n100_slow = read.csv("distances_hmc_d50_n100_slow.csv")
distances_bps_d50_n100_slow = read.csv("distances_bps_d50_n100_slow.csv")
distances_bk_d100_n1000_slow = read.csv("distances_bk_d100_n1000_slow.csv")
distances_zz_d100_n1000_slow = read.csv("distances_zz_d100_n1000_slow.csv")
distances_hmc_d100_n1000_slow = read.csv("distances_hmc_d100_n1000_slow.csv")
distances_bps_d100_n1000_slow = read.csv("distances_bps_d100_n1000_slow.csv")

#dataframe creation 
distances_d10 = c(distances_bk_d10_n100_slow$X0,distances_bps_d10_n100_slow$X0,distances_zz_d10_n100_slow$X0,distances_hmc_d10_n100_slow$X0)
distances_d50 = c(distances_bk_d50_n100_slow$X0,distances_bps_d50_n100_slow$X0,distances_zz_d50_n100_slow$X0,distances_hmc_d50_n100_slow$X0)
distances_d100 = c(distances_bk_d100_n1000_slow$X0,distances_bps_d100_n1000_slow$X0,distances_zz_d100_n1000_slow$X0,distances_hmc_d100_n1000_slow$X0)
distances_val = log(c(distances_d10, distances_d50, distances_d100))
distances = c("10", "50", "100")
d_val = rep(distances, each=40)
samplers = c("Boom", "BPS", "ZZ", "NUTS-HMC")
s_val = c(rep(samplers, each=10),rep(samplers, each=10),rep(samplers, each=10))

df = data.frame(d_val, distances_val, s_val)

#plot 
ggplot(df, aes(x=factor(d_val, level=c("10", "50", "100")), y=distances_val, fill=s_val)) + 
  geom_boxplot() + xlab("Dimension") + ylab("log-scale L2-distance") + ggtitle("L2 Distances to true mean low noise")+ labs(fill='Sampler') 
