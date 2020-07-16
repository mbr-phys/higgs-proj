# higgs-proj

_Research extension from [4th Year project](https://github.com/mbr-phys/cpviolation)_

## Useful links
- [flavio github](https://github.com/flav-io/flavio)
- [flavio webpage and docs](https://flav-io.github.io)
- [tutorial with some useful examples](https://github.com/DavidMStraub/flavio-tutorial)
- [pdg 2020 for world avgs and parameter values](http://pdg.lbl.gov/)
- [hflav 2019 for some world avgs](https://arxiv.org/pdf/1909.12524.pdf)
- [Crivellin 2019 for bsmumu WC stuff](https://arxiv.org/pdf/1903.10440.pdf)
- will add more in time if needed

## To Do

- [x] sorted leptonics 
- [x] Switch from manual running top quark calculation to using flavio's running 
- [x] Add LO B Mixing 
- [x] Add general B mixing from Crivellin - _need to fix some issues with it still_
- [x] Add R(D), R(Dst) 
- [x] Look into resolving Bsgamma SM value - _modified in flavio's files to make it possible, still much higher fits than previously though_
- [x] Add tree level semileptonics to fit - _same WCs as other tree levels so pretty simple_
- [ ] Does the likelihood function change if observables are correlated or not?
- [x] Look into `RKpi(P+->munu)`, it's going all funky right now - _doing the individual BRs instead seems to fix this_
- [ ] Conclude from heatmap results for CKM modification
- [ ] Finish adding to `world_avgs.yml`

## General Project Stuff

- [ ] Sort out WC language for observables not yet added
- [ ] Sort out obliques - do we need to add WCs to the basis?
- [ ] What do we do about extra parameters in fit when coming to SM4 etc?
- [ ] Higgs signal strengths, R(K)s
- [ ] Is it better to use `Likelihood` instead of `FastLikelihood`?
