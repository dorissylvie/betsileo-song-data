# Rapport de nettoyage du dataset v2

## Lignes en sortie

| Partie | Lignes | Versets / mots |
|---|---|---|
| train bible | 8385 | 4194 |
| train lexique | 296 | 148 |
| train segments | 4301 | 903 |
| train chansons | 394 | 197 |
| validation bible | 74 | 37 |
| validation chansons | 48 | 24 |
| test bible | 534 | 267 |
| test chansons | 66 | 33 |
| test lexique | 52 | 26 |

## Retraits

| Split | Raison | Lignes |
|---|---|---|
| test | chanson | 2 |
| test | mal aligne | 8 |
| test | manuel | 96 |
| test | religieux | 972 |
| train | chanson | 11 |
| train | doublon | 2 |
| train | fuite | 8 |
| train | mal aligne | 96 |
| train | manuel | 4 |
| train | paraphrase trop libre | 17 |
| train | religieux | 84 |
| validation | chanson | 7 |

### Termes religieux trouves (test)

jesoa (612), jesosy (414), andriamagnit (154), andriamanitr (138), kristy (102), mpisorogn (46), mpisoron (44), reference biblique (42), petera (42), tempoly (42), filazantsara (32), demonia (30), sinagoga (26), pilato (26), batisa (24), sabata (20), masina (18), fanahy (18), tompo (16), fitsanganana amin'ny maty (12), zanak'andriama (10), satana (10), fanahy masina (10), andrea (8), kaisara (8), soratra masina (8), soratsa masina (8), fagnahy (8), famonjena (8), iskariota (8)

### Termes religieux trouves (train)

fitsanganan'ny maty (24), nitsangana tamin'ny maty (14), fitsanganana amin'ny maty (12), famonjena (10), mpamonjy (8), hitsangagna amin'ny maty (6), nitsanganany tamin'ny maty (4), natsangana tamin'ny maty (4), nitsangagna tamin'ny maty (4), jehovah (4), hitsangana amin'ny maty (4), natsangagna tamin'ny maty (2), azy ô (2), mesia (2), natsangany tamin'ny maty (2), tafatsangana amin'ny maty (2), hitsanganan'ny maty (2), hitsanganany amin'ny maty (2)

## Validation : versets retires ou modifies

- **tsivahiny/velo-po.json:3** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:6** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:10** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:12** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:13** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:15** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **tsivahiny/velo-po.json:17** retire : chanson : traduction de Claude non relue, exclue de validation/test
- **2JN 1:1** (bts) : `Aho lohologna ro mamangy an'Andriambavy voafidin'Andriamagnitsa sy ny zanany amin'ny alalan'itoy taratasy itoy, izay tiàko amin'ny fahamarinana, eko tsa aho irery, fa izay rehetra mahalala ny marina koa` -> `Aho lohologna ro mamangy an'Andriambavy voafidy sy ny zanany amin'ny alalan'itoy taratasy itoy, izay tiàko amin'ny fahamarinana, eko tsa aho irery, fa izay rehetra mahalala ny marina koa`
- **PHM 1:15** (mg) : `Izany angamba no nialany vetivety teo aminao mba hanananao azy mandrakizay,` -> `Izany angamba no nialany vetivety teo aminao mba hanananao azy mandrakariva,`
- **2JN 1:2** (mg) : `noho ny fahamarinana izay mitoetra ato anatintsika sady ho ato amintsika mandrakizay:` -> `noho ny fahamarinana izay mitoetra ato anatintsika sady ho ato amintsika mandrakariva:`
- **2JN 1:2** (bts) : `noho ny fahamarinana izay mitoatsa ato agnatintsika sady ho ato amintsika mandrakizay.` -> `noho ny fahamarinana izay mitoatsa ato agnatintsika sady ho ato amintsika mandrakariva.`
- **Majeur 7/galeoka.json:1** (mg) : `sady manondro no manontany hoa aiza no tsara` -> `sady manondro no manontany hoe aiza no tsara`
- **Majeur 7/galeoka.json:6** (bts) : `aho e te hisotro galeoka e` -> `aho te hisotro galeoka`
- **Majeur 7/galeoka.json:7** (bts) : `io e ro ampy sady mba takatakatsy ny vola` -> `io ro ampy sady mba takatakatsy ny vola`
- **Majeur 7/galeoka.json:8** (bts) : `andao e lehe hisotro galeoka e` -> `andao lehe hisotro galeoka`
- **Majeur 7/galeoka.json:17** (bts) : `tandremo e lehe hisotro galeoka e` -> `tandremo lehe hisotro galeoka`
- **Majeur 7/galeoka.json:19** (bts) : `agnao e ja misotro galeoka e` -> `agnao ja misotro galeoka`
- **tsivahiny/velo-po.json:7** (mg) : `salama aby ve tsika ao` -> `salama daholo ve isika ao`

## Test : versets retires apres relecture


## Garde volontairement

Mots religieux devenus courants, gardes comme dans le train (decision de
`bible/remplacements_proposes.md`) : fiangonana, vavaka / mivavaka,
fivavahana, tompo (minuscule), lanitra (le ciel), Dada, sampy, fahagagana,
mpanoradalana. A retirer en les ajoutant a `HARD_PATTERNS` si besoin.

## Chansons

Decoupage PAR CHANSON (un refrain ne doit pas etre a la fois en train et en test).
Validation et test : uniquement tes traductions. Les traductions de Claude
(`traductions_claude.tsv`, jamais relues depuis leur creation) ne vont qu'en train,
et sont listees dans `chansons_claude_a_relire.tsv`.

| Split | Chanson | Paires | dont traduites par Claude |
|---|---|---|---|
| train | 5 tsindrina 5 | 16 | 0 |
| train | Asabotsy mifankahita | 15 | 0 |
| train | Ataovy serieux | 27 | 0 |
| train | Mihinana soa | 9 | 0 |
| train | Rapagnarakandro | 8 | 0 |
| train | agnao | 23 | 0 |
| train | autres | 6 | 0 |
| train | mlaolao | 10 | 0 |
| train | sesily | 18 | 0 |
| train | tsahavelako | 11 | 0 |
| train | tsiahyMisareba | 17 | 0 |
| train | zipety | 9 | 0 |
| train | ziva | 28 | 0 |
| validation | galeoka | 16 | 0 |
| validation | velo-po | 8 | 0 |
| test | Tagnitagny madigna | 20 | 0 |
| test | tsofyrano | 13 | 0 |

### Tes corrections reprises de `corpus_chansons_exclus.tsv`

Ce fichier a perdu sa ligne d'en-tete (`raison ref bts mlg valide`) : `build_corpus_chansons.py` ne lisait plus tes corrections. Elles sont reprises ici :

- oladad/5 tsindrina 5.json:0 : `ao, avia agnao ho agnambala` -> `avia ianao ho any ambanivohitra`
- oladad/5 tsindrina 5.json:40 : `soa fianatsa ro mahavokatsa liaka aaaa` -> `tsara fianatra no mahavokatra lehiretsy a`
- oladad/agnao.json:24 : `nivoendrivoendry aho, sy mba nanova hafa e! n tsa nahita aho! ny foko ro naratse!` -> `nivezivezy aho, ary mba nanova hafa e! nefa tsy nahita aho! ny foko no naratra!`
- oladad/agnao.json:30 : `mararirary an’ia` -> `mararirary aho`
- oladad/agnao.json:31 : `fa mila ho roso gegy an’ia!` -> `fa saika ho lasa adala aho!`
- oladad/sesily.json:4 : `fitiavana io raha tsa hifagnirena` -> `fitiavana io raha tsy hifanerena`
- oladad/ziva.json:50 : `eka roaba, avelao agny fa tsa magnahy igny` -> `eny re ry dada, avelao any fa tsy maninona iny`
- tsivahiny/tsiahyMisareba.json:21 : `sy mahita kilonga apela ko te handodotsa` -> `rehefa mahita zazavavy dia te hanenjika`
- tsivahiny/tsiahyMisareba.json:30 : `bozy e lanin’ny akoho ny vary an!` -> `lanin’ny akoho ny vary an!`
- tsivahiny/tsiahyMisareba.json:32 : `mandihana agna tsiena apitso hividianako zipo kely` -> `mandehana any an-tsena rahampitso hividianako zipo kely`
- tsivahiny/tsiahyMisareba.json:42 : `avognavogna tsa ho lefy tsendry e` -> `avonavona tsy ho lefy mihitsy e`
- tsivahiny/tsiahyMisareba.json:60 : `ny fitiavako an-drika ro mahafaty fa mandihana fa tsa mila koa` -> `ny fitiavako anao no mahafaty fa mandehena fa tsy mila koa`

### Traductions de Claude : 158 relues par toi (`training/chansons_relues.tsv`), 0 pas encore relues


### Tes traductions : points a verifier (laisses tels quels)

- Majeur 7/galeoka.json:19 : « misotrotro » : « misotro » ?
- tsivahiny/velo-po.json:2 : « apela » (betsileo) reste cote malgache
- Majeur 7/Ataovy serieux.json:27 : « fahatanitany » : mot a confirmer (aussi ligne 31)

Les corrections appliquees (erreurs nettes de Claude, fautes de frappe) sont dans
`modifications.tsv` et `MANUAL_FIXES` en haut du script.
