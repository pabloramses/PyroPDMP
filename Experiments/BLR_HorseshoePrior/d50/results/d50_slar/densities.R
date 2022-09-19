library(ggplot2)

hmc_c = read.csv("postSamp_hmc_d50_slar_run1.csv")
bk_c = read.csv("postSamp_bk_d50_slar_run1.csv")

hmc = read.csv("postSamp_hmc_d50_slar_run1.csv")$X14
bk = read.csv("postSamp_bk_d50_slar_run1.csv")$X14

df = data.frame(bk, hmc)





ggplot(df, aes(bk)) + 
  geom_density(aes(bk), fill="blue") 
  
  
png("densities.png", units="in", width=7, height=5, res=600)
ggplot(df, aes(hmc)) + 
  geom_density(aes(bk, fill="Boomerang")) +
  geom_density(aes(hmc, fill="NUTS")) +
  xlab("") + 
  ylab("") + 
  coord_cartesian(ylim=c(0, 3)) + 
  scale_fill_manual(name="", values = c("Boomerang"="deepskyblue", "NUTS" = "red"))+
  #guides(color = guide_legend(override.aes = list(linetype = c(0, 1, 2, 0),
                                                  #shape = c(16,NA,NA,NA), 
                                                  #size = 1)))
  
  theme(legend.position = c(0.165, 0.9),
        legend.background = element_rect(fill = "white"),
        legend.title = element_text(size=3),
        legend.text = element_text(size=17))
 dev.off()
 
 
