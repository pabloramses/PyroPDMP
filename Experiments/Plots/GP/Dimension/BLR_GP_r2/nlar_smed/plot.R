# plot by number of dimensions
library(ggplot2)

r2scores_bk_d10_n1000_smed = read.csv("r2scores_bk_d10_n1000_smed.csv")
r2scores_zz_d10_n1000_smed = read.csv("r2scores_zz_d10_n1000_smed.csv")
r2scores_hmc_d10_n1000_smed = read.csv("r2scores_hmc_d10_n1000_smed.csv")
r2scores_bps_d10_n1000_smed = read.csv("r2scores_bps_d10_n1000_smed.csv")
r2scores_bk_d50_n1000_smed = read.csv("r2scores_bk_d50_n1000_smed.csv")
r2scores_zz_d50_n1000_smed = read.csv("r2scores_zz_d50_n1000_smed.csv")
r2scores_hmc_d50_n1000_smed = read.csv("r2scores_hmc_d50_n1000_smed.csv")
r2scores_bps_d50_n1000_smed = read.csv("r2scores_bps_d50_n1000_smed.csv")
r2scores_bk_d100_n10000_smed = read.csv("r2scores_bk_d100_n10000_smed.csv")
r2scores_zz_d100_n10000_smed = read.csv("r2scores_zz_d100_n10000_smed.csv")
r2scores_hmc_d100_n10000_smed = read.csv("r2scores_hmc_d100_n10000_smed.csv")
r2scores_bps_d100_n10000_smed = read.csv("r2scores_bps_d100_n10000_smed.csv")

#dataframe creation 
r2scores_d10 = c(r2scores_bk_d10_n1000_smed$X0,r2scores_bps_d10_n1000_smed$X0,r2scores_zz_d10_n1000_smed$X0,r2scores_hmc_d10_n1000_smed$X0)
r2scores_d50 = c(r2scores_bk_d50_n1000_smed$X0,r2scores_bps_d50_n1000_smed$X0,r2scores_zz_d50_n1000_smed$X0,r2scores_hmc_d50_n1000_smed$X0)
r2scores_d100 = c(r2scores_bk_d100_n10000_smed$X0,r2scores_bps_d100_n10000_smed$X0,r2scores_zz_d100_n10000_smed$X0,r2scores_hmc_d100_n10000_smed$X0)
r2scores_val = c(r2scores_d10, r2scores_d50, r2scores_d100)
r2scores = c("10", "50", "100")
d_val = rep(r2scores, each=40)
samplers = c("Boom", "BPS", "ZZ", "NUTS-HMC")
s_val = c(rep(samplers, each=10),rep(samplers, each=10),rep(samplers, each=10))

df = data.frame(d_val, r2scores_val, s_val)

#plot 
png("r2scores_by_dimension_nlar_smed.png", units="in", width=5, height=5, res=600)
ggplot(df, aes(x=factor(d_val, level=c("10", "50", "100")), y=r2scores_val, fill=s_val)) + 
  geom_boxplot() + xlab("Dimension") + ylab("r2 scores") + ggtitle("Medium noise - Large size")+ labs(fill='Sampler') + theme(plot.title = element_text(hjust = 0.5), legend.position="bottom")
dev.off()