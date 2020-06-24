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

- [x] sorted leptonics as well, need to check with Alex I've done it right
- [x] Switch from manual running top quark calculation to using flavio's running 
- [ ] Add B mixing - _waiting on operator interpretation from Alex_
- [ ] Add R(D), R(Dst)  - _trying to find expressions using operators and how to describe the operators simultaneousl_
- [ ] Finish adding to `world_avgs.yml`
- [ ] Add `vev`, `lambda_QCD` and any other new nuisance parameters to the `parameter_x.yml` files in `flavio/data`

## General Project Stuff

- [ ] Sort out WC language for observables not yet added
- [ ] Sort out obliques - do we need to add WCs to the basis?
- [ ] What do we do about extra parameters in fit when coming to SM4 etc?
- [ ] Higgs signal strengths, semileptonics, R(K)s
- [ ] Does the fit include errors on all the parameters going in? 
- [ ] Is it better to use `Likelihood` instead of `FastLikelihood`?
