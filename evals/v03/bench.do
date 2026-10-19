* econkit v0.3 benchmark: each block is checked by bench.py against published / reference-package values
import delimited abdata.csv, clear
xtset id year
* [1] Stata manual xtabond example 1 (one-step)
xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984 year, lags(2) noconstant
* [2] two-step, Windmeijer-corrected
xtabond n l(0/1).w l(0/2).(k ys) yr1980-yr1984 year, lags(2) twostep vce(robust)
* [3] predetermined regressors
xtabond n l(0/1).ys yr1980-yr1984 year, lags(2) twostep pre(w, lag(1,.)) pre(k, lag(2,.)) noconstant vce(robust)
* [4] Roodman (2009) system GMM, one-step robust small
xtabond2 n L.n L(0/1).(w k) yr*, gmmstyle(L.(n w k)) ivstyle(yr*, equation(level)) robust small
import delimited mpdta.csv, clear
xtset countyreal year
* [5] Callaway-Sant'Anna, never-treated, no covariates (R did / differences)
csdid lemp, ivar(countyreal) time(year) gvar(first_treat)
* [6] not-yet-treated, outcome regression with covariate, event-study aggregation
csdid lemp lpop, ivar(countyreal) time(year) gvar(first_treat) notyet method(reg) agg(event)
gen rel = year - first_treat if first_treat>0
gen never = first_treat==0
forvalues k=4(-1)2 {
  gen F`k' = rel==-`k'
}
forvalues k=0/3 {
  gen L`k' = rel==`k'
}
* [7] Sun-Abraham
eventstudyinteract lemp F4 F3 F2 L0 L1 L2 L3, cohort(first_treat) control_cohort(never) absorb(i.countyreal i.year) vce(cluster countyreal)
gen D = first_treat>0 & year>=first_treat
* [8] de Chaisemartin-D'Haultfoeuille dynamic
did_multiplegt_dyn lemp countyreal year D, effects(4) placebo(3)
import delimited wb.csv, clear
* [9] wild cluster bootstrap (G=12)
reg y x z, cluster(g)
boottest x, reps(9999) weight(webb) seed(1)
boottest x, reps(9999) weight(rademacher)
import delimited darfur.csv, clear
* [10] sensemakr (Cinelli-Hazlett 2020 Darfur)
sensemakr peacefactor directlyharmed age farmer_dar herder_dar pastvoted hhsize_darfur female i.village, treat(directlyharmed) benchmark(female) kd(1 2 3)
import delimited wb.csv, clear
* [11] two-way clustered OLS, wild bootstrap on one dimension (reference: brute-force enumeration, 4096 refits)
reg y x z, cluster(g yr)
boottest x, bootcluster(g) reps(9999)
