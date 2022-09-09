def summary(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "5%": v.kthvalue(int(len(v) * 0.05)+1, dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95)+1, dim=0)[0],
        }
    return site_stats

def variance_limits(samples):
    site_stats = {}
    for k, v in samples.items():
        site_stats[k] = {
            "up": v.mean(0) + 3*v.var(0),
            "low": v.mean(0) - 3*v.var(0),
        }
    return site_stats