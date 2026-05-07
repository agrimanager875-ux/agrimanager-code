# Upstream Citations

This file records the external projects, simulators, data services, and
software frameworks that should be cited when describing AgriManager
experiments. AgriManager does not claim to have created WOFOST-Gym, CycleGym,
Cycles, DSSAT-Gym, DSSAT-PDI, DSSAT-CSM, PCSE, WOFOST, VERL, NASA POWER, or
their underlying data.

## References

## Citation Guidance Summary

| Tool/Simulator/Data | Citation source | Recommended action |
| --- | --- | --- |
| CycleGym wrapper | CycleGym repository does not provide a formal citation. | Cite the CycleGym software repository when needed and cite the underlying Cycles simulator paper for the native simulator. |
| Cycles simulator | Penn State/Cycles publication. | Cite Kemanian et al. (2024). |
| WOFOST-Gym | WOFOST-Gym documentation citation section. | Cite Solow, Saisubramanian, and Fern (2025). |
| DSSAT-Gym / DSSAT-PDI | DSSAT official citation guidance and DSSAT-Gym software repository. | Cite DSSAT recommended references, mention the version used, and provide the DSSAT-Gym software URL. |
| DSSAT-CSM | DSSAT official citation guidance. | Cite Jones et al. (2003), Hoogenboom et al. (2019), and Hoogenboom et al. (2024) as relevant. |
| PCSE | PCSE documentation/repository does not provide a dedicated paper citation. | Cite the PCSE repository and cite WOFOST model references where PCSE is used as the WOFOST implementation path. |
| WOFOST model | Wageningen/WOFOST model references. | Cite de Wit et al. (2019) and, where appropriate, van Diepen et al. (1989). |
| VERL / HybridFlow | VERL GitHub citation and acknowledgement section. | Cite Sheng et al. (2024), HybridFlow arXiv:2409.19256. |
| NASA POWER data | NASA POWER referencing guide. | Include both the POWER project acknowledgement and data reference with service name, version, and access date. |

### CycleGym

The CycleGym repository does not provide a formal citation. Cite the software
repository when discussing the wrapper, and cite the Cycles simulator paper
below for the underlying native simulator.

Software: `https://github.com/kora-labs/cyclesgym`

### Cycles Simulator

Kemanian, A. R., Shi, Y., White, C. M., Montes, F., Stockle, C. O.,
Huggins, D. R., Cangiano, M. L., Stefani-Fae, G., & Nydegger Rozum, R. K.
(2024). The Cycles agroecosystem model: Fundamentals, testing, and
applications. *Computers and Electronics in Agriculture, 227*, 109510.
`https://doi.org/10.1016/j.compag.2024.109510`

Software/releases: `https://github.com/PSUmodeling/Cycles`

### WOFOST-Gym

Solow, W., Saisubramanian, S., & Fern, A. (2025). WOFOSTGym: A crop simulator
for learning annual and perennial crop management strategies. arXiv:2502.19308.
`https://arxiv.org/abs/2502.19308`

Software: `https://github.com/Intelligent-Reliable-Autonomous-Systems/WOFOSTGym`

### DSSAT-Gym / DSSAT-PDI

The optional DSSAT-Gym/DSSAT-PDI path uses `gym-dssat` version `0.0.8`,
`gym_dssat_pdi` commit `63f2c529e0bd339b4553beb9aa56d56af83b5e2b`, and
`dssat-pdi` version `4.8.0.24_2`. Cite the DSSAT model references below and
mention these runtime versions when describing DSSAT-backed experiments.

Software: `https://gitlab.inria.fr/rgautron/gym_dssat_pdi`

### DSSAT-CSM

Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D.,
Hunt, L. A., Wilkens, P. W., Singh, U., Gijsman, A. J., & Ritchie, J. T.
(2003). The DSSAT cropping system model. *European Journal of Agronomy,
18*(3-4), 235-265. `https://doi.org/10.1016/S1161-0301(02)00107-7`

Hoogenboom, G., Porter, C. H., Boote, K. J., Shelia, V., Wilkens, P. W.,
Singh, U., White, J. W., Asseng, S., Lizaso, J. I., Moreno, L. P., Pavan, W.,
Ogoshi, R., Hunt, L. A., Tsuji, G. Y., & Jones, J. W. (2019). The DSSAT crop
modeling ecosystem. In K. J. Boote (Ed.), *Advances in Crop Modeling for a
Sustainable Agriculture* (pp. 173-216). Burleigh Dodds Science Publishing.
`https://dx.doi.org/10.19103/AS.2019.0061.10`

Hoogenboom, G., Porter, C. H., Shelia, V., Boote, K. J., Singh, U., Pavan, W.,
Oliveira, F. A. A., Moreno-Cadena, L. P., Ferreira, T. B., White, J. W.,
Lizaso, J. I., Pequeno, D. N. L., Kimball, B. A., Alderman, P. D., Thorp,
K. R., Cuadra, S. V., Vianna, M. S., Villalobos, F. J., Batchelor, W. D.,
Asseng, S., Jones, M. R., Hopf, A., Dias, H. B., Jintrawet, A., Jaikla, R.,
Memic, E., Hunt, L. A., & Jones, J. W. (2024). *Decision Support System for
Agrotechnology Transfer (DSSAT) Version 4.8.5*. DSSAT Foundation.
`https://www.DSSAT.net`

Data repository: `https://github.com/DSSAT/dssat-csm-data`

Spack resource tag: `v4.8.0.28`

Resolved data commit: `c1a31af16fc82659e0b024d58e40b021981b182b`

### PCSE

The PCSE documentation/repository does not provide a dedicated paper citation
for the software. Cite the repository and cite the WOFOST model references
below when PCSE is used as the WOFOST implementation path.

Software: `https://github.com/ajwdewit/pcse`

Use this alongside the WOFOST references below.

### WOFOST

de Wit, A., Boogaard, H., Fumagalli, D., Janssen, S., Knapen, R.,
van Kraalingen, D., Supit, I., van der Wijngaart, R., & van Diepen, K. (2019).
25 years of the WOFOST cropping systems model. *Agricultural Systems, 168*,
154-167. `https://doi.org/10.1016/j.agsy.2018.06.018`

van Diepen, C. A., Wolf, J., van Keulen, H., & Rappoldt, C. (1989). WOFOST: a
simulation model of crop production. *Soil Use and Management, 5*, 16-24.
`https://doi.org/10.1111/j.1475-2743.1989.tb00755.x`

### VERL

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H.,
& Wu, C. (2024). *HybridFlow: A flexible and efficient RLHF framework*. arXiv
preprint arXiv:2409.19256.

Software: `https://github.com/verl-project/verl`

### NASA POWER

NASA POWER requires both a general project acknowledgement and a data reference.
The WOFOST weather pools use NASA POWER meteorological data through the
WOFOST-Gym/PCSE weather-provider path.

POWER project acknowledgement:

> The data was obtained from the National Aeronautics and Space Administration
> (NASA) Langley Research Center (LaRC) Prediction of Worldwide Energy Resource
> (POWER) Project funded through the NASA Earth Science/Applied Science Program.

POWER data reference template:

> The data was obtained from the POWER Project's Daily X.x.x version on
> YYYY/MM/DD.

Fill `X.x.x` and `YYYY/MM/DD` with the exact POWER Daily service version and
the access date used to generate the released weather caches. Do not substitute
the paper submission date unless it is the actual data access date.

POWER references:

- Referencing guide: `https://power.larc.nasa.gov/docs/referencing/`
- Daily API documentation: `https://power.larc.nasa.gov/docs/services/api/temporal/daily/`

## Draft BibTeX

Use these as starting points for the paper bibliography. Check formatting
against the final venue style before submission.

```bibtex
@misc{cyclegym_software,
  title = {{CycleGym}},
  howpublished = {\url{https://github.com/kora-labs/cyclesgym}},
  note = {Software repository. No formal repository citation is provided upstream.}
}

@article{kemanian2024cycles,
  title = {The {Cycles} agroecosystem model: Fundamentals, testing, and applications},
  author = {Kemanian, Armen R. and Shi, Yuning and White, Charles M. and Montes, Federico and Stockle, Claudio O. and Huggins, David R. and Cangiano, Maria L. and Stefani-Fae, Giovana and Nydegger Rozum, Renata K.},
  journal = {Computers and Electronics in Agriculture},
  volume = {227},
  pages = {109510},
  year = {2024},
  doi = {10.1016/j.compag.2024.109510}
}

@misc{solow2025wofostgym,
  title = {{WOFOSTGym}: A crop simulator for learning annual and perennial crop management strategies},
  author = {Solow, William and Saisubramanian, Sandhya and Fern, Alan},
  year = {2025},
  eprint = {2502.19308},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2502.19308}
}

@misc{gym_dssat_pdi_software,
  title = {gym-{DSSAT}-{PDI}},
  howpublished = {\url{https://gitlab.inria.fr/rgautron/gym_dssat_pdi}},
  note = {Software repository. AgriManager optional DSSAT runtime used gym-dssat 0.0.8, gym_dssat_pdi commit 63f2c529e0bd339b4553beb9aa56d56af83b5e2b, and dssat-pdi 4.8.0.24_2.}
}

@article{jones2003dssat,
  title = {The {DSSAT} cropping system model},
  author = {Jones, James W. and Hoogenboom, Gerrit and Porter, Cheryl H. and Boote, Kenneth J. and Batchelor, William D. and Hunt, L. A. and Wilkens, Paul W. and Singh, U. and Gijsman, A. J. and Ritchie, J. T.},
  journal = {European Journal of Agronomy},
  volume = {18},
  number = {3--4},
  pages = {235--265},
  year = {2003},
  doi = {10.1016/S1161-0301(02)00107-7}
}

@incollection{hoogenboom2019dssat,
  title = {The {DSSAT} crop modeling ecosystem},
  author = {Hoogenboom, Gerrit and Porter, Cheryl H. and Boote, Kenneth J. and Shelia, Vakhtang and Wilkens, Paul W. and Singh, Upendra and White, Jeffrey W. and Asseng, Senthold and Lizaso, Jon I. and Moreno, Luis P. and Pavan, Walter and Ogoshi, Richard and Hunt, L. A. and Tsuji, G. Y. and Jones, James W.},
  booktitle = {Advances in Crop Modeling for a Sustainable Agriculture},
  editor = {Boote, Kenneth J.},
  pages = {173--216},
  publisher = {Burleigh Dodds Science Publishing},
  year = {2019},
  doi = {10.19103/AS.2019.0061.10}
}

@misc{hoogenboom2024dssat,
  title = {Decision Support System for Agrotechnology Transfer ({DSSAT}) Version 4.8.5},
  author = {Hoogenboom, Gerrit and Porter, Cheryl H. and Shelia, Vakhtang and Boote, Kenneth J. and Singh, Upendra and Pavan, Walter and Oliveira, F. A. A. and Moreno-Cadena, L. P. and Ferreira, T. B. and White, Jeffrey W. and Lizaso, Jon I. and Pequeno, D. N. L. and Kimball, B. A. and Alderman, Phillip D. and Thorp, Kelly R. and Cuadra, Santiago V. and Vianna, M. S. and Villalobos, F. J. and Batchelor, William D. and Asseng, Senthold and Jones, M. R. and Hopf, A. and Dias, H. B. and Jintrawet, Attachai and Jaikla, R. and Memic, E. and Hunt, L. A. and Jones, James W.},
  year = {2024},
  publisher = {DSSAT Foundation},
  url = {https://www.DSSAT.net}
}

@misc{pcse_software,
  title = {{PCSE}: Python Crop Simulation Environment},
  howpublished = {\url{https://github.com/ajwdewit/pcse}},
  note = {Software repository. Cite together with WOFOST model references where relevant.},
  url = {https://github.com/ajwdewit/pcse}
}

@article{dewit2019wofost,
  title = {25 years of the {WOFOST} cropping systems model},
  author = {de Wit, Allard and Boogaard, Hendrik and Fumagalli, Davide and Janssen, Sander and Knapen, Rob and van Kraalingen, Dirk and Supit, Iwan and van der Wijngaart, Rob and van Diepen, Kees},
  journal = {Agricultural Systems},
  volume = {168},
  pages = {154--167},
  year = {2019},
  doi = {10.1016/j.agsy.2018.06.018}
}

@article{vandiepen1989wofost,
  title = {{WOFOST}: a simulation model of crop production},
  author = {van Diepen, C. A. and Wolf, J. and van Keulen, H. and Rappoldt, C.},
  journal = {Soil Use and Management},
  volume = {5},
  pages = {16--24},
  year = {1989},
  doi = {10.1111/j.1475-2743.1989.tb00755.x}
}

@misc{sheng2024hybridflow,
  title = {{HybridFlow}: A flexible and efficient {RLHF} framework},
  author = {Sheng, Guangming and Zhang, Chi and Ye, Zixuan and Wu, Xun and Zhang, Wang and Zhang, Ru and Peng, Yu and Lin, Haibin and Wu, Chuan},
  year = {2024},
  eprint = {2409.19256},
  archivePrefix = {arXiv},
  url = {https://arxiv.org/abs/2409.19256}
}
```
