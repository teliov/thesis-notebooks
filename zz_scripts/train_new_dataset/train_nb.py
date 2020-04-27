import argparse
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, make_scorer
import joblib
import os
import numpy as np
import pandas as pd
import json
import sys


def train_nb(data_file, output_dir):
    columns_interest = "LABEL,GENDER,RACE,AGE,004d2a1bcadec845497c95a165b4fa4ba673cc051a2f58114483cea9,01acc83bcd6be176213c84ca7fab158e89604248fd8c2284d0de46ff,01b7281c9d3dd1247867eb69139bff96dd1b95e43b3ba62adedb644a,02431a478e53e3c5d3066c8e75ad6a589a2582b1716753fb617c1423,025f3b10ae50e9c3f4791b969ea541edeb6ab5a1749fd5793026ca15,02859d17c4d0e24628c516b499aaa12022c47addb2c9af7f483390c2,029201fd3f91841dc4ca225d2ad45f94beaf2d06db5a178dcce6f0f9,02c0f56b1824ed7adaa714f76f69da0c77f60c85fabf7e6fedb969ec,03238e17f270e876d4638b2ec6289ec7b9c3d1ef08da5d305d0049ca,033c50c5dd2a1827e4ab715eb072f4dc72eb152451653cd2ace30470,039bca5a6ad5106c1e0a445b50945ac1c4ce2af184709bf37e90d99f,04d688fc065c8ae55a11c4e3d16222149a78878253d00a238bef69cb,05d4b39421ea256faca648529c294df00ccd76d0de812e9787dda8fb,05fd53b9e6d3a9c91969e66b8f1a88ff2e5bb7f80f9f70dffca41f27,06cb3d2b6c43de91ddf0c01522af6ff5454e887490dbbb01d9e4bba4,06f41fa65a3abd353222402d4ecdb227013f1a2cf991dc264eadfa44,07f4fcdd4474efdece9b0036d3d8b45849b8691ccdd3d30f3f13c08a,09203f180a4eaa195ed35065ec3705c56b21012cd9b261cf5c8fe891,09423838df994f3681bc39b825513db0b0d2edcbfa706599173017bb,095308e97a67fe60579acd2156dd26d5bb4f808b6ad48b79d5493b01,0c7f6feb86aac8d207612b68cd99669c463df4a6236fe5442b631954,0cbfc8283b3a0f93663673251e9e64943cacc8fe2248c1a0aee5e66c,0d3bb8cb718d7c1ff59546b031668a872d988ff3ef2828b9e0b6477c,0d8a205870ddb381454a0a40d885ea736d70e0525e6581799371ae5f,0df780436d68b44d03ff08ef12a43dcca5b9f8fd4e87ce1e5c90263b,0e08e38481da65a4d2bc55cb58a6219f6ac82aea7aaf4c3fdf4ce7b8,0f4472ecb6a7ba8179a20470e735dd6b8a078c17f6e2f7021cda6553,0fe76a40dd9752fb6a500a717e30ef91548f90509958cf89219c55ce,10d3045ab5f6b28962f66eb142f026e29440061788e3e070bd64c094,10da1352ef47642b9d2a173c2984f85f175ba479fdfe22ab2fc77e91,10ec4543dc934be02d47b2122a49fe259702abed27f911338a9815c5,12755935957a4a323cecb130d66d0a680cff5ef81ba24a6d3e207027,1310d177602475cbdec2965e7b82220450b18d061610c4dfbc165c39,13f0347d45dcb52be8e1a7106877695a1d190eab69abc907ebca2602,1434a6c33b7a2bb8d95c2c88c2028e5030f018db2b5da0307a01448a,146cbb323984eaa8a6a8eae776027df278ca775e72e3b5a09ca2fc27,14ba1f9cbcb905d6cd007b82ad1ae88fad4843e6c8f65009ed79f603,1668e3836033166076708dd0bdd4d2dc8457f9d46c8c675be4d03541,1686b0fc1e02eb0e1164cfe51c4f81f0025d57bc3f2f124133a84f98,16c962bc1abd3c2557f4d0e94e3dc1c98f63b06f9b31a924380a8566,16f017cd17bc6573e35610984e0a362e0dce7cd47663a9dfdb7c5c59,17063a25660ca0be4eed7f24241d9dfc7a752ac1b7814918c9c34afb,18ca73772057ae60dd016540e0d9f43dce2e37f3166c4f160e1363aa,192a07d43a6ccaa88d2e4915a22422335137d568ef9489deb5fc4cd8,192c24d1d4fa9718b47dce9ee949b9ebaf113d75ac61d210a0e69a91,193873ffce538fecef3eba4245ae74c2d79f0485bd93c73386109eed,19d22ac8d256464618c63a1420df4e96492fb94c789cccaa54de685a,1a0cc8f425259fae5e3e6dae3874a54001a908c03158aed8e49a6920,1a466ba76e18a6848e9148e852f9c60156e2fb67a164144c91a44d5f,1b1f54cbd95bfc714772f25c7b5838d918fc4f5a22940763cbae7937,1b925045c3f98c2e3250786586fa98d05a29f24a8bf9f8a551b68098,1c041889187bf4d3a2a4d3e07aaeca7f4c4cc535002bc398aef4c422,1c78fee1c5b508d51402828e93510e6901828b7cf31fc11f06778587,1cdb6ff2b029aeb60d937151875510fe5c525ab52d37b9ab95a9d1b2,1cebbff2518142ae65b17786c874279ba6596421377b9433fab04ffc,1d12575e23e797261a1f6c215d94a778fb1b2b3abfec1cce81e43514,1e1b6d3262ffb2d957e82aa3b53e852f39dfd7ada2e4222542f07b59,1eb24b0e1a3aad2f99aaa1cc9132cfc4918ebf15d2ec0bf6c3dc5c29,1ed8eee352fcd4f149dda8fb8bee7a17910cac2bed12b7b3e7f5ae4f,1fb2f7eb2813c5f10a1a5f9f460e010fefede0e7594a57c1705ecc43,2054ff037d718b1f1258030069a8ea978b08fe16aeb44a001f5d582b,205662e865df56ca3acd4caeb082c75976cb68dfe7b04f5a9128a823,20f078d591c3938f30fbd6218625301214da8dfdd0c6a9cde58cc80d,2109510432761a316b5258b7a349a5634dd47ca8f048c10607f64934,212a9650524fcaac378c1bd175f37a240e2cbd8718317c634530f963,21a0b655dc700939e019c597c63e6825b4f92228df9586527a432171,223a397ed58b1a7362b363f5e141aa9d855178559f6f8d60f31d27f4,2287055583c0f1f51a8b13906a81ec1104159d1c457d465511aeada6,22d34b7fa25d9421e77484f2367619a33230b4d591fb658a7a64c0cb,22f2fdab55812dbe357ef34580ebb59a9afb710e7e49b48aa928d1f0,23622ad88b655992661dcf9a38d4569d790bddbc7870f42aa228b9da,23aa4195d51eb504527500d89651b974ff771aa140a2b204c631682f,245d02d2fea8bbb59628ab117f8c88895132cddacae82c2278f38a2c,261dcc2046e2ab046e8cb37ac4f4c28c85aec8c245965b2febc8c34a,26f1c9e6e8263ab35d31deb65e12b6c58e0a0e23ae6dbecc76cc0b6d,2735031db23bfbf71286400d727049f91ab06c1f4f9e5f61cacb5854,2799686bdf34d2a3b6f792b24696e1fd2f27c80b20bd4c9d40944f5e,28788987af3c56116bd79e5ca5810ab083750b3ce0849bcc84c58a7c,28bf5d81761a883d6e26f5bafce33028968b0ac3528f19369133f04c,28c0837a435b3248465e6e2801a85a8068efe1e84539b1cd0f01a993,28e950697c1413441d21f94cfbbdedae923175db9d2007463c510fd8,2a2d873eca10a3e9b5b18f215baedcd62081c786f9e487aebdb7c1cd,2afff4efa1230f4b951520555275fe3fccb43b40649ee8878cf89945,2b28ad7e2ef84375a5746e1447d2fb0f37f1e68a9250fdcd9635695c,2beec2e26ae6415f1ff1743af570af53cc8ad90db5688c65cd4e21ff,2befb2e3a7cd839b89c611a70cf7b9a77e8c24bee1f140ae38de3ad0,2d2a0420160206b7ae29712a6e7bb9f6f0aa4efccc75af8330cd219d,2e7ad354e85c4d6e7a3ace6413235f8afc626422d9be28671e97fe14,2f66ceec3c15d5417295dce834664e4234b0421693f5609838487129,2fe6d99eb788392eb93382757b162bfcab7936d47c789f5e20f00de6,312c07bdd360ce7efbc62f51172e6bb299e64819176a61cb3cda5d8a,32a10b9112926d09aa574495201c25e53b3468d8ff5134ccfb5ff3d0,32f427ab99bce716982b3d7820f9ffecc6d4970fc88da9fa5729da48,35724cd7c9d10129f0d3b79a9aef0e3e26743e2378832a074e034e61,363aa29ab96f062ee4495640a04b55b908ec79b29e6289fcec8910da,36aa13639a906a81d4e9e327856a8b6e4b984a5b8adfc1ed2b68a54b,36c572503a8f8b783f0da9493c35a8770c9076d6506c63b6eef37e71,37a30bfb1314420ca5ad2ccc24ea77f3eec4dd136949caed5d175007,389ecbd4673a240798c23a80ef278bc55f0e9b2fc01d004010addc33,396c8dd9318b65be69765899b3134778054bbb8a342a467633e3b168,3989456997fe172a7349c0b3676b71c0376b7bb84812666ed51b2260,39e701dcc5971e85af8f6e502826be444f061f1013ef3a32ae4f98dc,3b117c6435324926c1ade3e3a3d0799577f8f66677ea194a1fbdbc4f,3ca78b21ceb10d94f111682530f05c59501c2bb6856d84705ef2d2dc,3ccf25410a97805252ba29fb9417f47651df7da7b5bf8c79ba1f46dc,3d676c91076e0dbb8126367c220d446a29146476e82dc01e26aad757,3e7166f6b9c7f979748fa0e224454da80a0b3d4d608a1422f5caed8f,3e7a9e10d6391a011d698886bb273c87d9343fca97f74cbd797a0eaa,3edaddcdacedfdae957de03af9d59590e550ae84790f647c77bf2f77,3ee6a7b5a1d5da3a9a069e262afdb0afc5955a2577d477d5fe13a279,3fb832b201f13088f92d8791bd918471622d0090662f1e3ec60da957,3fc36ae1744b6c87e4477862186c7e9ffa7df6f651175687b7768ac6,3fda6393decd7eaef9733d1ab144c8c194b01ee5e32dc8c8324e73cb,3fe7475349164c17ff52305f12defdcc0460b971adaf12dbcd7d4554,40ec99bf7189c7abde851a3a491f7df312f008ecef8fa64237a0743f,410cce1e38d1e8210346ff9ac3ac4da622f1d579574b3ce57a7248a2,41d22270eeb1d6d801e691b428d34060bceb589197f263dc28b303b8,4375ac659b40c3f1b2e110fc6604f3a930e72b16783e165271f082c7,43c6f3ec3da518d61e93715a55524f66e4fb085314d74e6208b6c89a,441e1f49a3ac39357369f2018bd4cdab3c8cbd79379be80a2550aa10,446d2abab108476afe90edcaa8be1934c8a5471ea7dc591d4a0b7ba5,44a1037a8a49395b152add4595ca1e6cb134f491f0dc9c376f0e3519,44c13a28422eebfffbdc9eccd72925136a1ec488019534291b013c46,476bad35e8a32319824e7a0653397cce0437e215b7c077b8bb7edb09,49094b9e5fe8168296bf7185a69c80d25593cb0987ecc5abce580a21,494588229efd3c7c08fbb212c8c77e3c69461275697efd3040d58391,49bfe114716bbf692a4ca505ea1d670904fc828d477ad7bd6d7bb43e,49d55a1583e5c5c2b5f641e21ebe344dae9b12de07aceec7adf0e109,4a650b565bcd79880b14c7c31a0ec7e664c86913ed0765c301c77794,4cd751518d9df57df5e353a8d1e1c1eeb338f970dc6b7353f28bea86,4faf1a1b087d0c059cc5d09ac771027c7f8d24179448a75a8e7b4d02,500d581fe1745f9bfc7605bd608b9e1b6d17951d274b6ac0609ac815,503e6cd2b22fa20d874008383762ecdf9240579bcdfcde1a7bef3e33,517a7ce26d6758b18a58a5ca01d20a1f72e39ffc9cc87fbfe84765e2,51f0f756661f5ffed5b98073f5c0938fa42495103ead1b6ceb802765,51f89bfb2b2ae18320e806c5150ced732ee30cd23c4c09d284e78db8,52332094d2bf70f9d53b1747b6afa6db5fe595617fc0455d14a47b8d,52440a63a9dfb2bda8737abadc9a02189dd6bcb2122a677be2acb3c5,5245c4c611c357e6424560a56026d422d3eaa759792925d3502b1811,52f3a3bc07de76212047ef199d68fc1934b595c7e0a73a11d2bf7785,53c1bc2734fababe36b03a5e100dc30967fd5f72cb3304f50ee5cead,546eb8f1cb9689d1bc3955d92051f66b431f64522f1c4683923d47bc,559ef64e8a95766073112c84d8c583ce2e0a2244ec129b85eb0e3bfc,5735a067988a40cacdab9d1b8a5d148e5a739979634a5f0974122f17,57444fd0a992d7215f1d3ad288f2155f12e657d3441fe14304934961,578cf375931714cd6e15e58438a0250d2088d84c3f927500eae9ed83,58233edcb20064476119c687347f4d1abbdf6410d0dba09ba4dacf7b,5843a004a3813353de2787bbf2686ec6284cc8cc035111e546401c44,59bfe7254d927e37c493e2f0fb50ab39999c42ec709e86cf7b235f72,5c94f181def3220a5ab03fdcc58150df5cacfb332acb0c647c18c49c,5d4a0dcf90c7bc96217df9bb300e6b3686ed4b73d9bc8736b5b42012,5ead93159784c6c67f7c040bcc322d53ca8ed0359b6986f448665b22,5ee0ad6c3b25e177a90a0d98909d6d12ab8fb546fdf64c4c8f761f5e,619774facd5899d35dd246076889e2445bc588a513310d0e7e903510,61c8faf4628bc43c4daf53f9aa17b546bdad37900f8d8896ec742f37,62a086c34b1cf0ebd1c7e78fb1411fe6ba011ef1f0a5037c47c45fde,63a9eea4be08b44d2a55851478530318d0e7434c2a89007eb9109ddd,65cf27f0990fc011a797a3657ef5602d968d838f304feeb82900e148,677af1cf3d8d0abd1a8c9773286289b5a731ea7c8ae677567414cc17,67ada96a7541e82ce8e0091f11fd76cb620e6a026dab8f371b50d4d4,67fe1b0607dced2d78d47eb7b8f2b599c0823043d54f0d875d9e5505,69da782bec6e0561cd02bb9b6fb1242010ce034adab4ef0569d32463,6a77a6ea87ea954cdf3daa4bca52c47a3033aacfa45947d2e234c212,6afca9564539b6a4f109cac854d21404db02df2d75503abdb1680bd0,6b1103328178b0eb33ea0793691afaf85e0658214e37f8d43db784c5,6b82087334ae1889825af76d0d775ac9daa7fe3ae692186347a3eec3,6bd28415c30cdde6de027574aa79c61dbfd3335349a9f11feb741d84,6c08d0a0b34a31f00367f163d595dc42cae10f4793b66ad89d600936,6e5679e114a5650a48fd9620496e66033d4f5cca4933106a4f60b75a,6f1896060c7e7c5e0665bcb390850cfc19d4e47866a358900f4d8c37,6f2bab2bb64b131e2cf5381d63cfd3442512326992cc4b27df837104,6fe1a8d0c4d3e025b6b5e340a3136f8db64305a5899e702908a20be4,6ffb720e222bd78000c5bc78d067a71c2545d594430394b95c1031c6,703cf77867ac879e0da8b007eefe9d407476a0cdaf1132c11066081f,71d03f6bcb415d02dd4cb37dc68d6c2834e072ed821e4e82b47de516,71dfae102ec00c4d9f960ad62244511f5f6441e944c9ed3f064addcd,720c101c0090a923cfd7cb6dc62bb00abbe2dfaefee42d4e4fb1422b,72a054b2b7aca4530e13f7f4886982f99f51a5deda8793b8927a43a0,73069be93b329f19709df9e5fadcbbb3b35e1e1553b3f3aebac891b3,733e1e0c619b7a9d2ea15d9f56f20c94869921cb7aa75b52c33ccdd3,7460a1df01d0013b74f2f39b872535f18559ed4747b67da3273553a1,74662d3d2ede018a2a4b417fc04fbff969b937c9b60c5bda367b8e07,74fa86ff958cece3d32d7747904c6340fb7b7262cc7e91d75324f2b7,751cf0c0b031c70d33facce81fe1351d40441b0fce96dc16c9010d76,752d4bbaf61d07cf5dd6c98e09c6035c7e97eea91a7c768f453cf09f,7660052ddb046f2a6929ab6d9d1b252cb20ed208e32c13deeb735fe4,767c9388b3593960cfeb75145c4e0061ae941425fd584c33a47aca84,76c0b503ad830cb80a2ba2ec4fc9a394845b73b9a34d9e277d8c3b33,76dd9d1c37202d59185d7caa72fe76a58360dcc13b872131f8251ebd,7725196c5cd5bfaa7daa59e350ce4002cfdbc9f160eac3250f1f5086,7734c15b6d3ec41729a3ede59cef7922a4d69dd651cf54cafe5b5596,7757e67bd399478fd93d37dbc1b7da8d0ed11bd382408e6d2ae7ba51,7939396ee9d774533247df2fecc317d7a6bb368b9247b4bb07610c01,79d88ad5021f4a50c11220dae6feaf3bbabc54eead0899275dda0300,7aaa74616407718e7da9c22b463d64c8a4069dad0ef56d6f76dba0ac,7b0b2636641584c3a2cde33f116273f8caa8fee7103813d8ddfd5a13,7b152bae63092dc826ed8ff7ac1fad003cf37635b4be106769c41a75,7b62bf9743dc17cbaf97defe93af62cf2e1305ecf066bc5e14cda672,7c20f829a771219ece0f8f1fc531994a18d66bed3550fea645578243,7c6ef1def409846265af9e52a2e5755aae0c88d9fc07696a02683ae6,7d063801e12b750d1176a339c4a5eebf4a39554e7502734e1355ec3d,7d30a2ed5b07597e7b960c90158b9606dfb1cc3eb069087168455645,7d42c0d21238d5e39c6e548a4ce77fbb3f270fa9e38fbc7b45261984,7e57aeb27ba9f3c2aa46f975fe12aaf70280fe2ac712f4f302de3ad2,7ef8a43094d9870156c558ea6ee74a5c11658f7ce22017b1db6345d3,7f3485491251b954876484d21aed9c72824abad6f1fac5d1b7d66d8c,807e4b82f825ff066513ea2ef7869766e83b70f5ad23c1f4f45d4dba,811e76e05728742ba772c6d9b0828138c7c16b25ba394b2e6d456bd0,814cedacea819158682c930c78298928073ac20e89f6a99e2f739274,81b62ba806ca80e75a21509e749e86e8a65b3bac3f28a0dbdcc183b1,823d4e156af3c8c233fadb9826ab378cb440b9013953f0d66be10e5c,826e6ed0530a1223f5fa8a030f66db758ef3eabcccb2f47488847192,83f630151c43e0ffd7f10e50ec2f7372a73b1170d3fcf6f98223bf05,84232a120a6de53ead1ea4bbaa8e301318c4ff37c735b3631af1aea6,84385df32f44838d15c319e2534be581668c98a3c20fb3f82c20ad6f,85429958ca8501b9d1c63c75fbb8bcc9819d2eadadd95c4f3de3e1b4,858332075a98c33310e5de37dc4290c3552a07c39e394631e011a5fe,85d949b144be61c5263fec420bcb2a4eaf8bece63dce2a5c54cfeaef,85e3b85b581eae22180c8cf5ae744166e70f1b2002da60a8e1c53e97,85ec196b9fcfd32cbdee5a044b20d7c406ff7846cb02be6803c93ba2,861dc93ed5cd1c489df17a6c447b0af95cae81eedcd4e0e97e7fa0fe,8666a538fa7d128654725f1fa34e6c44a6091ed458d644d5bbfd753a,87f0ba940716ae5cf22b6453690625a81328c2a0d75ad2900d0c35b0,88e3734465cfa73e79558a79b395d4492318ce1decdecd7409af6134,88f4773871f52c171fb974858cb6f9f14e1791bd04a1cab1cca3b5c2,89f713b2759b5edb8ae11d88971f3ffd20f53ae9815e033be732dfba,8b057821b72bb077e84930b6184ce51e09f431128c5b181708bc5250,8bf4510ea2ee8610f4e55c4263431ea2c857572bffba5fe265dc1d2b,8cf862138b224c9f4faf038e00d33cb3e2a05735fab114191f3b7db0,8d070eaa03bc73971c82d88354dcf0d6729962f71c7357e8182bc0f5,8e2bb984769cf137429d4e251d7a90803937c8e893584aa41ca5798f,8eb2c0dc03b4ef7e4460373006802e341612e7a8978eddc661a88737,8ecefd608d13429302287e01f56ca67ffe8800643c021a8080ea291d,8f9834e5d9929f19ff42c014b79bf6fa07b39e2897402f865a7c4e38,8ff93d77a9bd298030bd0b4f4c7652d75606b210166b0dcb5a9259e5,9185a2117fe1e255b8d5b6393157a116dd49719c401ef804be792251,919b32644afa4bdc8fc9b91bcf9430cc9c7cbd4842d47658810250ae,921789127595258319f67eb0b8cd618a572006e641165e03b13bfb32,9542bcd1800f19b013cc7c69979763cfed1c27692c050d39e6bdf422,95c876c850c2ab7706368023480e0d40e18f152153dac9ade991f5c7,95f80eabeb5f350bf8ea0d6b2331936864fd539668570b5e27116d6e,96d27c966e5668667571d9889cafc94b62dcd10c0b2ebbc66b3513c6,97aa40c7e0f28851a6e357dfcfb49f08e820d932a911cd06a127468f,97bd50b353dba728d06a17cfee6cb99847543ec35a8384713c48a777,98f6fac90b90f659c2a0ae623cd913c6f88d298a0c604970cdbc98e2,998a2fd0eec3e43d1f359aa26714291cbcfe3e4c0bd14de81522801b,9a7532cbd861e99907b57e2c69c2886b7d7b70899967d5d6eecac892,9ab6356a959b4124fc2c9282252dc63ff9d1d79b896c6f6bd71ffe31,9b0bdaa4d4d1e7e5c28e8cff38aa8872a83362d7d2757b1f8d3dfdee,9b5117697d7a0b8d17d6dba88ab5ec2dd8b870a812e28654811ffc49,9be824ac6f12dd72421468c16e86cfe33f7fdfa4875c37a5eea2186f,9d62267c63e590c40f31090c22d9a81159882fcca1d06884ec96552a,9ddf38bbb42835924a1acd338b091e5c9d0e92259981b4c47ea3d71a,9e92a664bbfb42fddcfb860748e7db35963ff4264ecf32add4b9f095,9ea5e0adce9d91365905c1e6a1accd553fea13d5962dc45e0a1d0260,9eb8ed00fea5c191a71cb08a6bb2dbd95445b22f17ced2056ae61bb9,9fc95fa89a9f8756ae27b6824034373d88652b09a934eebaf7ec1c49,a016ad0231d121e4ce3554bec6665852fff70cb7ced803753900b330,a1858289add24287e512494c4169e6fed611b9331a6d250e96863a73,a1b4c06c3ea6ab83d933b1980b6738a683aec3825b3f73a390c0c741,a1ca593b8000dbb3c48c28be5f4546baa998198420ffb9792e7116f3,a1e497bd0957305231d83d4455132d90991c916c5b96735c3ea786b5,a2617ffd6e36ecb07e82cc458db5a3b374a746aada9b40b393a787b0,a39c85681ce14f8c3244957a41c6ab3be77cccbc2dcaa71439000876,a55ad2ed39538a110b8e7216daec93308956d25f07952279207b8c4b,a6981f923d2b3b43b421c34281bc4358b7a75109a37bd26d229db0f6,a6a908f6a79186ad3a452c8b49e6503e3845737bfe423222bdfe0ed5,a6ef13dfe9a6b03e2c5518cc76d473affc4c1bd66de7b2767461b525,a7a40aa9ed26a3fd02669c3faf31de0f36ec9f4f9cbc20dfcb604fc5,a7f9c921a60e5c8f5a8b590da7dfb0e60fadef6b600e5984a482063f,a8155822acce246d816ca52daf54bc704279712140e806be1df6fc62,aa767f3d9127d6d8b06c149a87fdf6897b55e517caded15ebd273379,ab31b8e3e90e987abe0c2d35484ce77469288e834f4ba45a8ca5e768,abf7aa440e6366d4557078f58f7d121205f7ac614411393195a9e383,abffb97bd90761124d1420bd86bc45bbae0411de416408e7c6851829,ac51d117e705cf95ff09aafaa06a6dfd52ac3abae595931aff59cdcf,ae59ce5da84c436e2aa728070c13b809b69b108814a244efbae26825,aeb52985328d38f37279d063ae81f0bcdca2232c7468b5f69591a7cb,b044a578d8e246ae53087fd86f13c3bebf4bd1e22da9483e3a3aecfc,b0558619808b0e3d7a7347c3739cf5bc2f863c327348384b3911fa03,b17ba219479e72cb9472a031b4a4cecb42861730d1301cb824861f1b,b223042009d6494a2c645370387e95c7936ea89cc5d606afcc2a5a8b,b24185e5c7952c0edad74e61463b2677c4443f11d52de92c25df98d1,b2e6283fc2fd75907247409f23150b1a24ef4a75c5f45d628b42d135,b3410497bde6908c297cfc9286cd3535815465032f432a72b59be041,b41321a6ba6cb6844e7ddc395f952f97919ff673cc1b40737e3e2f63,b4b566d519722b9d15464c7b0d78ba643f85d0d144da43a666dc8094,b59bc8b87dce84e50068a94192aea4d6407c60beabe3574db1c92153,b7bdef76df722c5b4a8f5290397030bc11917cedf6144d1a65a1f161,b806e16e2ab10ec5e97bf8391b1f6b713098add2e72d172c242e7f0b,b8d8ff2f66a697be8a313c0bdf014bc440a63d1e60c0be70d93d5652,ba432f340cbf704e64908fdca758f4e6322a4766f7651f31e1a8b5c2,bb1814759ece3b328489385602f7769942b45360c23f4926dfdd905c,bb1af8d5b2f3b8f94618dd7b8911e8c6db9298c90747acfeec6b0f84,bd8e11ef139c39a058c6c384b668304a40598ee80cae269bd40afee8,c01c488a8935d973bb726ac058ee9a4aa95d1b44a84828551215f324,c13451c794d93a9ff20ef5c240089bdea3f25391024c7d1344fdf56c,c13c111201d30c7dcfc5f69b223bdd770570e3a4d96102a9270ed013,c1c5b5b89624154064582c5446654eedcfa057e2cf7f64b658beef71,c216af590839f5afb50cca4bc10c2d2a34e7565dd785631ca96d41c2,c2d432674b4fe5c5e2a914e7f31613700af7a2de51c82f75ce3cb298,c4ab2db625b32c4a1af0fd9d43586fc4f121779824958c2ecc400a33,c56f4122cf02e81fc3f8ccb80fa1b009f9dd6e0f974564f0baf6b574,c6ffd19f126272d62f35a91a51e54b3e374d628f29e958b07eab95f3,c7156270815d7d0c8ae7151c9625e9f814bc4fc64412f5ffc894e47a,c7a16eea751073bb478a39e8f7b2706268c0209621a33e5d7c2313b4,c80736bb7ffb8adfbda7ee6b4ff22c919db839ddf91e555f4778b0ee,c81b416ef8f0534f78c40a2896efdb8590c6e63a774c51200f3e6a92,c85f1942bfa61983437cb3c8e44ebd826055d28f9e7c724c73ee7a2e,c88324cd84dc4d7ef6198ca95e1891f6bd811ee8ab3b3680f24211d1,c9f28460e4f5c75bff95f0bbf400dbff74e1a0e7059a57191d9537b5,caacb3fdac9aa2d615c7fceeb41495a5dfda5555ac67225bd75465b6,cb8566370a69304c91f04fa4cdeae618bcb6e6412b1af6a054bc9988,cd6e438379d68f1991771072c5fc8c4e725c6f70d0db83b3cf13bdd7,ce09928e8c37eaec2e58bd757d511a46dbd168c7c1e2026161ce4a35,cee13b9ebb7bb92362d1a2c22b4f681637588d39c5c20498027eb58c,d093675157fbff9298b5c51408e66417cd677f5a944ec1361d5edfc0,d24dd581b92a567e912356e381205bef6445f4b14e36b9079e2406a9,d2e283dc93026af352a141bbdd4d0f81a8863fbb9134ca29d9ee128b,d5c478569b1beb99112d0c234c879131df55bd838d8e40fb4fc3dd4c,d5f92aa53423af2c05cad73c9f2507289cb9930cefc097463714329f,d6194a5a150068af0a3a51387850e5fffc8fd65147491b762a21ccc9,d61d6abd7e7d2d3f9cad104a348a752c4a6b091b18c3f259ac13c520,d7c0ea72732278558c9a03c6f9be91b43fae4c51d1f5f355a98bdab0,d8713e210924d464bed545bf76fe488b83f47061e0bd756b820b837c,da019821bdd6e2f2f218d167eb784c9b6ae5862d2496ab439682f29c,da40e27287ddac4ae4f52c7148dd9d9141e216783d72a8640c0f52f7,dca266a296a44fe65deec0f53b303d0d8e5ab2c0b39c98031aef3f46,dcf3e42632c477781e24de52d9b8216dcea08ae00b6d03198af426fb,dd050a1057e988465bb012a58ded9cbbbafdb45fbcb7600ca544b17b,dd52980213ed3f58007375b494cf13182420dd104acf39cb84c683ab,dd78b4226ffe45e4532188e723fe867e4ffe72c1c7890931b8b9c5a1,de6819bf5dfc10650586dedfaee9abfad3f18e238a3a52cbbc5be3ab,de70dd9ec4b81ad0b2e841e0e118f2b04d6e1996de4402642fce6ae8,ded06d7c33248317975b8bef8f52f270714db6b126a96a684e5f5c79,df0f3dc0392f9176943bbe96c2a0ec8eb87bfa36d9f1e63689a4dd04,e06a51ae1d927eab36899a2f6e85c58a7db96684250784f0f5a8bc03,e0bf72b1e83e13dbc5cc8db5595a6fa6c8a81416372e90571bf16ca4,e119b3e5d79a3d8375dcf913bafd96a64199eadcff708fc6d55002b6,e2ee96458b4c405592d20b231e4b1a6e508f92fcfa4062d77a6bba71,e63f20ea0b9ced31e9f46abf863908a9bf9da37b62e025c48a515647,e70692ec7ac252aa1663696f60958383725e3d214990d6634b7f0211,e8782d2de9ad4883f790fbd92e2312370c1e9593616e9e47ab88a48d,e8ab05a92798047e11c91912bcfc71aabe3f1531809b0631673b5212,e9190b5443ac1fb07154398db57f75adf126412d0cc42fa30e799ce5,e9ddb1ca0dacbae8003c3eed12f89557c4d7f3dd1482ee2e068c873b,ead4dca6864a1e84017c3da2813cf035da5b7f652b1423e3ec0fc82f,ec07aea60ba80c5a3263f95235c15e7b06eb503b0e7f1bbd9ae06c3c,ed999058533c037cb6bd3069ccd2d4b533023ceb40acb5d316bb5730,ed9cd1792b97cc04e6a07b96251ddaf3158838e3ec31655cce03eab9,ee14c150596724922748adf6086361be8c03dbfc0702c4c8c0c9b521,eff5873b3091cbd4b9e86c684c7f0297e5a2518b8d6e4cab54e3943c,f1398095a0b97a5279ba912b8059ef50ff6af32d32f9b48c0386e668,f1a5fe4562ba7a58da2d59ddfdf57ad29029aeaa064d580c0caecddc,f1e386f5773e030aa2451368ccc79679145f798b75fa38179b2e1bce,f37e6c589b8ff5a6594b06f3265a6bedf559e4611aabc122d4ef1ed2,f383eb73fae3dc352f9c5f1cad62e1e7743d7fb61deab077a22e2b82,f46c12c92d2faaf1a289e2d42d281b825b3b6571f4c7eb742f0f043a,f46d8931cee66d4ee907a8e51c9fd62d3a6b6d85072264c9e3d1bb75,f5ecf602935a123a7e103b26aad93463635fbe83e2e6e126341c0562,f74873162867f7014ff45760e57397a2dc1ec47eb26907b95e997ea9,f84bc83cd39d5c35f4dd04b4fc303d1f9ef93de38ab23e5d37944389,fac32e56366418f0589aeff45906d08f2bd5ebfdf2f25193eb20bb12,fb7381e7a00b832c27efacf4b80e751166212e5e23c8f7006dc5a865,fb8baf3d208e5b7fb8ab8a09d92de4d459f01c538b72aac81c9e21b3,fb8e059b5629b6190a92884d3dae262b5af0a7647deade7e425724ee,fc53919f7ed62152dc002caaf693014461ded3a1b1b0a7ff88f224c3,fc91229b566e3c4d36bead9a18fe11de98cf48c1828f83db5a0783ca,fcc89ce7fae7c2ee3426e27e1c67a9d564343b96b5c39a2b8f0412c2,fce3a9d81b8660d5a6d3ed09f4a801d0b35045777842a450d2b0c04d,fd42985ea19c7dffd6b22d18c4279f3a1c9e2193f01d6d2a6f1ad944,fe45433788530e5bd233c20eaeabd5320f009b02830a4b22767d9308,fea33294cd211320d85c8b1af2448da5dfbef79aa5dcc3cbd51fabb4,ff15d27632aef66d31fe6cc964de80ddcbcc849a17028a64a935ba87,ff2d30810017aabbfbbbed53bf6235269516c18aa7b0ca34f054d0a6,ffc9a960b1f8f1d4709d5c9b9b8be078efabaa6a74b4577404424762"
    columns = columns_interest.split(",")

    dtypes = {
        'LABEL': np.uint16,
        'GENDER': np.uint8,
        'RACE': np.uint16,
        'AGE': np.uint8
    }
    symptoms = [item for item in columns if len(item) == 56]
    dtypes.update({item: np.uint8 for item in symptoms})

    symptom_gender_clf = BernoulliNB()
    race_clf = MultinomialNB()
    age_clf = GaussianNB()

    df = pd.read_csv(data_file, usecols=columns, dtype=dtypes)

    class_map = [[age_clf, ["AGE"]], [race_clf, ["RACE"]], [symptom_gender_clf, ["GENDER"] + symptoms]]
    valid_labels = df.LABEL.unique()
    nb_clf = models.ThesisNaiveBayes(class_map, valid_labels)

    y_target = df.LABEL
    X_data = df.drop(columns=['LABEL'])

    num_jobs = os.cpu_count()
    scorer = make_scorer(accuracy_score)
    results = cross_validate(nb_clf, X_data, y=y_target, scoring=scorer, cv=5, n_jobs=num_jobs,
                   pre_dispatch='n_jobs', return_train_score=True, return_estimator=True, error_score='raise')

    # save the model with the highest test score
    test_scores = results['test_score']
    train_score = results['train_score']
    fit_time = results['fit_time']
    score_time = results['score_time']
    estimators = results['estimator']

    avg_test_score = np.mean(test_scores)
    avg_train_score = np.mean(train_score)
    avg_fit_time = np.mean(fit_time)
    avg_score_time = np.mean(score_time)

    # best score
    best_idx = np.argmax(test_scores)
    best_estimator = estimators[best_idx]

    # save results
    train_results = {
        "avg_test_score": avg_test_score,
        "avg_train_score": avg_train_score,
        "avg_fit_time": avg_fit_time,
        "avg_score_time": avg_score_time,
        "best_test_score": np.max(test_scores),
        "best_train_score": np.max(train_score)
    }

    train_results_file = os.path.join(output_dir, "nb_train_results.json")
    with open(train_results_file, "w") as fp:
        json.dump(train_results, fp)

    estimator_serialized = best_estimator.serialize()
    estimator_serialized_file = os.path.join(output_dir, "nb_serialized.joblib")
    joblib.dump(estimator_serialized, estimator_serialized_file)

    return True


if __name__ == "__main__":
    file_path = os.path.dirname(os.path.realpath(__file__))

    module_path = os.path.join(file_path, "../..")
    module_path = os.path.abspath(module_path)

    if module_path not in sys.path:
        sys.path.append(module_path)

    from thesislib.utils.ml import models

    parser = argparse.ArgumentParser(description='Medvice Naive Bayes Trainer')
    parser.add_argument('--data', help='Path to train csv file')
    parser.add_argument('--output_dir', help='Directory where results and trained model should be saved to')

    args = parser.parse_args()
    data_file = args.data
    output_dir = args.output_dir

    if not os.path.isfile(data_file):
        raise ValueError("data file does not exist")

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    train_nb(data_file=data_file, output_dir=output_dir)
