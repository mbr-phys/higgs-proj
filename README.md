# higgs-proj

_Research extension from [4th Year project](https://github.com/mbr-phys/cpviolation)_

## Useful links
- [flavio github](https://github.com/flav-io/flavio)
- [flavio webpage and docs](https://flav-io.github.io)
- [tutorial with some useful examples](https://github.com/DavidMStraub/flavio-tutorial)
- [pdg 2020 for world avgs and parameter values](http://pdg.lbl.gov/)
- [hflav 2019 for some world avgs](https://arxiv.org/pdf/1909.12524.pdf)
- [Crivellin 1903 for bsll WC stuff](https://arxiv.org/pdf/1903.10440.pdf)
- [Similar research being done in GAMBIT](https://arxiv.org/pdf/2007.11942.pdf)
- [New values for B(s/d) -> mu mu](https://indico.cern.ch/event/868940/contributions/3905706/)
- will add more in time if needed

## To Do

- [x] sorted leptonics 
- [x] Add LO B Mixing 
- [x] Add general B mixing from Crivellin - problematic near mH+ ~ mt(mt) in loop functions 
- [x] Add `R(D)`, `R(D*)`
- [x] Look into resolving Bsgamma SM value - _modified in flavio's files to make it possible, still much higher fits than previously though_
- [x] Add tree level semileptonics to fit - _same WCs as other tree levels so pretty simple_
- [x] `RK` and `RK*` need added, uses same WCs as `Bsmumu` so simple enough - _essentially no constraint from these, why?_
- [x] C7,8 contributions from Crivellin to `RK`, `RK*`, and `bsgamma` 
- [x] Conclude from heatmap results for CKM modification - _might come back to this, but happy with conclusions that modification is negligible, see ckm_notes.tex_
- [ ] Fit in wrong sign limit
- [ ] Is there a reason `R(D)` and `R(D*)` are fitting simultaneously fine? They historically do not
- [ ] Does the likelihood function change if observables are correlated or not?
- [ ] Finish adding to `world_avgs.yml`

## General Project Stuff

- [ ] Sort out WC language for observables not yet added
- [ ] Sort out obliques - do we need to add WCs to the basis?
- [ ] What do we do about extra parameters in fit when coming to SM4 etc?
- [ ] Higgs signal strengths
- [ ] Is it better to use `Likelihood` instead of `FastLikelihood`?
