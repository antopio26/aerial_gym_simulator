## Benchmark.py - physics only (new code)
docker compose -f docker/docker-compose.yaml run --rm   aerialgym python aerial_gym/examples/benchmark.py
WARN[0000] Found orphan containers ([mini_aerial_gym_newton_x86_64]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container docker-aerialgym-run-693d28c42a90 Creating 
Container docker-aerialgym-run-693d28c42a90 Created 
/bin/bash: /opt/conda/envs/aerialgym/lib/libtinfo.so.6: no version information available (required by /bin/bash)
Importing module 'gym_38' (/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /opt/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
PyTorch version 2.4.1+cu121
Device count 1
/opt/isaacgym/python/isaacgym/_bindings/src/gymtorch
Using /root/.cache/torch_extensions/py38_cu121 as PyTorch extensions root...
Creating extension directory /root/.cache/torch_extensions/py38_cu121/gymtorch...
Emitting ninja build file /root/.cache/torch_extensions/py38_cu121/gymtorch/build.ninja...
Building extension module gymtorch...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] c++ -MMD -MF gymtorch.o.d -DTORCH_EXTENSION_NAME=gymtorch -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/TH -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/THC -isystem /opt/conda/envs/aerialgym/include/python3.8 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -DTORCH_MAJOR=2 -DTORCH_MINOR=4 -c /opt/isaacgym/python/isaacgym/_bindings/src/gymtorch/gymtorch.cpp -o gymtorch.o 
[2/2] c++ gymtorch.o -shared -L/opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o gymtorch.so
Loading extension module gymtorch...
Warp UserWarning: Python 3.9 or newer is recommended for running Warp, detected sys.version_info(major=3, minor=8, micro=20, releaselevel='final', serial=0)
Warp 1.10.1 initialized:
   CUDA Toolkit 12.8, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA GeForce RTX 4080 Laptop GPU" (12 GiB, sm_89, mempool enabled)
   Kernel cache:
     /root/.cache/warp/1.10.1
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
[12330 ms][__main__] - WARNING : This script provides an example of a rendering benchmark for the environment. The rendering benchmark will measure the FPS and the real-time speedup of the environment. (benchmark.py:24)
[12330 ms][__main__] - WARNING : 


The rendering benchmark will run by default. Please set rendering_benchmark = False to run the physics benchmark. 


 (benchmark.py:27)
[12330 ms][env_manager] - INFO : Populating environments. (env_manager.py:74)
[12330 ms][env_manager] - INFO : Creating simulation instance. (env_manager.py:88)
[12330 ms][env_manager] - INFO : Instantiating IGE object. (env_manager.py:89)
[12330 ms][IsaacGymEnvManager] - INFO : Creating Isaac Gym Environment (IGE_env_manager.py:42)
[12330 ms][IsaacGymEnvManager] - INFO : Acquiring gym object (IGE_env_manager.py:74)
[12330 ms][IsaacGymEnvManager] - INFO : Acquired gym object (IGE_env_manager.py:76)
[isaacgym:gymutil.py] Unknown args:  []
[12331 ms][IsaacGymEnvManager] - INFO : Fixing devices (IGE_env_manager.py:90)
[12331 ms][IsaacGymEnvManager] - INFO : Using GPU pipeline for simulation. (IGE_env_manager.py:103)
[12331 ms][IsaacGymEnvManager] - INFO : Sim Device type: cuda, Sim Device ID: 0 (IGE_env_manager.py:106)
[12331 ms][IsaacGymEnvManager] - CRITICAL : 
 Setting graphics device to -1.
 This is done because the simulation is run in headless mode and no Isaac Gym cameras are used.
 No need to worry. The simulation and warp rendering will work as expected. (IGE_env_manager.py:113)
[12331 ms][IsaacGymEnvManager] - INFO : Graphics Device ID: -1 (IGE_env_manager.py:120)
[12331 ms][IsaacGymEnvManager] - INFO : Creating Isaac Gym Simulation Object (IGE_env_manager.py:121)
[12331 ms][IsaacGymEnvManager] - WARNING : If you have set the CUDA_VISIBLE_DEVICES environment variable, please ensure that you set it
to a particular one that works for your system to use the viewer or Isaac Gym cameras.
If you want to run parallel simulations on multiple GPUs with camera sensors,
please disable Isaac Gym and use warp (by setting use_warp=True), set the viewer to headless. (IGE_env_manager.py:128)
[12331 ms][IsaacGymEnvManager] - WARNING : If you see a segfault in the next lines, it is because of the discrepancy between the CUDA device and the graphics device.
Please ensure that the CUDA device and the graphics device are the same. (IGE_env_manager.py:133)
Not connected to PVD
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
[13223 ms][IsaacGymEnvManager] - INFO : Created Isaac Gym Simulation Object (IGE_env_manager.py:137)
[13223 ms][IsaacGymEnvManager] - INFO : Created Isaac Gym Environment (IGE_env_manager.py:44)
[13356 ms][env_manager] - INFO : IGE object instantiated. (env_manager.py:110)
[13356 ms][env_manager] - INFO : Creating warp environment. (env_manager.py:113)
[13356 ms][env_manager] - INFO : Warp environment created. (env_manager.py:115)
[13356 ms][env_manager] - INFO : Creating robot manager. (env_manager.py:119)
[13356 ms][BaseRobot] - INFO : [DONE] Initializing controller (base_robot.py:26)
[13356 ms][BaseRobot] - INFO : Initializing controller no_control (base_robot.py:29)
[13356 ms][base_multirotor] - WARNING : Creating 256 multirotors. (base_multirotor.py:32)
[13357 ms][env_manager] - INFO : [DONE] Creating robot manager. (env_manager.py:124)
[13357 ms][env_manager] - INFO : [DONE] Creating simulation instance. (env_manager.py:126)
[13357 ms][asset_loader] - INFO : Loading asset: quad.urdf for the first time. Next use of this asset will be via the asset buffer. (asset_loader.py:72)
[13358 ms][env_manager] - INFO : Populating environment 0 (env_manager.py:181)
[13496 ms][robot_manager] - WARNING : 
Robot mass: 0.2500003944120692,
Inertia: tensor([[8.4501e-04, 0.0000e+00, 3.3087e-24],
        [0.0000e+00, 8.4501e-04, 3.3087e-24],
        [3.3087e-24, 3.3087e-24, 1.6900e-03]], device='cuda:0'),
Robot COM: tensor([[0.0000e+00, 0.0000e+00, 1.3411e-14, 1.0000e+00]], device='cuda:0') (robot_manager.py:441)
[13496 ms][robot_manager] - WARNING : Calculated robot mass and inertia for this robot. This code assumes that your robot is the same across environments. (robot_manager.py:444)
[13496 ms][robot_manager] - CRITICAL : If your robot differs across environments you need to perform this computation for each different robot here. (robot_manager.py:447)
[13511 ms][env_manager] - INFO : [DONE] Populating environments. (env_manager.py:76)
[13520 ms][IsaacGymEnvManager] - WARNING : Headless: True (IGE_env_manager.py:442)
[13520 ms][IsaacGymEnvManager] - INFO : Headless mode. Viewer not created. (IGE_env_manager.py:452)
*** Can't create empty tensor
[13521 ms][warp_env_manager] - WARNING : No assets have been added to the environment. Skipping preparation for simulation (warp_env_manager.py:190)
[13523 ms][asset_manager] - WARNING : Number of obstacles to be kept in the environment: 0 (asset_manager.py:32)
WARNING: allocation matrix is not full rank. Rank: 4
/opt/aerialgym/aerial_gym/control/motor_model.py:45: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(self.min_thrust, device=self.device, dtype=torch.float32).expand(
/opt/aerialgym/aerial_gym/control/motor_model.py:48: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(self.max_thrust, device=self.device, dtype=torch.float32).expand(
[13967 ms][control_allocation] - WARNING : Control allocation does not account for actuator limits. This leads to suboptimal allocation (control_allocation.py:48)
[15100 ms][__main__] - CRITICAL : i -99, Current time: 2.7706491947174072, FPS: -9147.297344188311, Real Time Speedup: -91.47287898549683 (benchmark.py:82)
[15513 ms][__main__] - CRITICAL : i -49, Current time: 3.1832783222198486, FPS: -3940.5848692971513, Real Time Speedup: -39.40581622779651 (benchmark.py:82)
[15609 ms][__main__] - CRITICAL : i 1, Current time: 0.0018620491027832031, FPS: 137009.2923312492, Real Time Speedup: 1368.1725586136595 (benchmark.py:82)
[15707 ms][__main__] - CRITICAL : i 51, Current time: 0.09990811347961426, FPS: 130671.96972350747, Real Time Speedup: 1306.6822806938005 (benchmark.py:82)
[15842 ms][__main__] - CRITICAL : i 101, Current time: 0.2348465919494629, FPS: 110094.38576308696, Real Time Speedup: 1100.9293282629044 (benchmark.py:82)
[15946 ms][__main__] - CRITICAL : i 151, Current time: 0.3395822048187256, FPS: 113831.91999084486, Real Time Speedup: 1138.30801132586 (benchmark.py:82)
[16045 ms][__main__] - CRITICAL : i 201, Current time: 0.4379606246948242, FPS: 117488.28728224296, Real Time Speedup: 1174.8758375113912 (benchmark.py:82)
[16141 ms][__main__] - CRITICAL : i 251, Current time: 0.5340738296508789, FPS: 120311.50367462094, Real Time Speedup: 1203.1085917949456 (benchmark.py:82)
[16250 ms][__main__] - CRITICAL : i 301, Current time: 0.6427583694458008, FPS: 119882.15249689348, Real Time Speedup: 1198.8166335640612 (benchmark.py:82)
[16344 ms][__main__] - CRITICAL : i 351, Current time: 0.7367458343505859, FPS: 121961.58743581687, Real Time Speedup: 1219.6119276251857 (benchmark.py:82)
[16433 ms][__main__] - CRITICAL : i 401, Current time: 0.8264245986938477, FPS: 124216.08717993989, Real Time Speedup: 1242.157288276416 (benchmark.py:82)
[16536 ms][__main__] - CRITICAL : i 451, Current time: 0.9290146827697754, FPS: 124277.10927720228, Real Time Speedup: 1242.7682223392471 (benchmark.py:82)
[16624 ms][__main__] - CRITICAL : i 501, Current time: 1.0169634819030762, FPS: 126116.09074540871, Real Time Speedup: 1261.1582464533308 (benchmark.py:82)
[16710 ms][__main__] - CRITICAL : i 551, Current time: 1.1033449172973633, FPS: 127843.33959709037, Real Time Speedup: 1278.4300809607332 (benchmark.py:82)
[16829 ms][__main__] - CRITICAL : i 601, Current time: 1.2220385074615479, FPS: 125900.66413910202, Real Time Speedup: 1259.0044307234157 (benchmark.py:82)
[16920 ms][__main__] - CRITICAL : i 651, Current time: 1.3126707077026367, FPS: 126958.95690097813, Real Time Speedup: 1269.587032493524 (benchmark.py:82)
[17008 ms][__main__] - CRITICAL : i 701, Current time: 1.4011971950531006, FPS: 128072.87896345592, Real Time Speedup: 1280.7270462794763 (benchmark.py:82)
[17097 ms][__main__] - CRITICAL : i 751, Current time: 1.4905707836151123, FPS: 128981.05013446254, Real Time Speedup: 1289.8088508981618 (benchmark.py:82)
[17185 ms][__main__] - CRITICAL : i 801, Current time: 1.577986478805542, FPS: 129947.35141515206, Real Time Speedup: 1299.4711581100155 (benchmark.py:82)
[17274 ms][__main__] - CRITICAL : i 851, Current time: 1.6667053699493408, FPS: 130710.20938482737, Real Time Speedup: 1307.1005980291518 (benchmark.py:82)
[17367 ms][__main__] - CRITICAL : i 901, Current time: 1.7600486278533936, FPS: 131050.49853329691, Real Time Speedup: 1310.5032101132817 (benchmark.py:82)
[17451 ms][__main__] - CRITICAL : i 951, Current time: 1.8445866107940674, FPS: 131983.58318184922, Real Time Speedup: 1319.8334435305148 (benchmark.py:82)
[17536 ms][__main__] - CRITICAL : i 1001, Current time: 1.9294803142547607, FPS: 132810.56293588647, Real Time Speedup: 1328.1043164915225 (benchmark.py:82)
[17629 ms][__main__] - CRITICAL : i 1051, Current time: 2.021613836288452, FPS: 133089.38078073572, Real Time Speedup: 1330.8919243106975 (benchmark.py:82)
[17720 ms][__main__] - CRITICAL : i 1101, Current time: 2.1136021614074707, FPS: 133353.00762866583, Real Time Speedup: 1333.5281207716346 (benchmark.py:82)
[17813 ms][__main__] - CRITICAL : i 1151, Current time: 2.206010580062866, FPS: 133569.3191634874, Real Time Speedup: 1335.6917480651614 (benchmark.py:82)
[17899 ms][__main__] - CRITICAL : i 1201, Current time: 2.2921934127807617, FPS: 134131.49462756928, Real Time Speedup: 1341.3136906483815 (benchmark.py:82)
[17989 ms][__main__] - CRITICAL : i 1251, Current time: 2.3822004795074463, FPS: 134436.68463064343, Real Time Speedup: 1344.365097180033 (benchmark.py:82)
[18077 ms][__main__] - CRITICAL : i 1301, Current time: 2.470229148864746, FPS: 134827.652130019, Real Time Speedup: 1348.2753501221803 (benchmark.py:82)
[18175 ms][__main__] - CRITICAL : i 1351, Current time: 2.5676729679107666, FPS: 134695.93095769593, Real Time Speedup: 1346.9579338065153 (benchmark.py:82)
[18261 ms][__main__] - CRITICAL : i 1401, Current time: 2.654592990875244, FPS: 135107.43844874972, Real Time Speedup: 1351.0730496980295 (benchmark.py:82)
[18354 ms][__main__] - CRITICAL : i 1451, Current time: 2.7468295097351074, FPS: 135230.52400841832, Real Time Speedup: 1352.304066317434 (benchmark.py:82)
[18445 ms][__main__] - CRITICAL : i 1501, Current time: 2.8385627269744873, FPS: 135369.6324842808, Real Time Speedup: 1353.694960439556 (benchmark.py:82)
[18541 ms][__main__] - CRITICAL : i 1551, Current time: 2.934239387512207, FPS: 135317.96073588007, Real Time Speedup: 1353.178397898778 (benchmark.py:82)
[18633 ms][__main__] - CRITICAL : i 1601, Current time: 3.026432752609253, FPS: 135425.25220972102, Real Time Speedup: 1354.2513485504114 (benchmark.py:82)
[18727 ms][__main__] - CRITICAL : i 1651, Current time: 3.1199088096618652, FPS: 135470.41910898368, Real Time Speedup: 1354.7030523242731 (benchmark.py:82)
[18818 ms][__main__] - CRITICAL : i 1701, Current time: 3.2112016677856445, FPS: 135605.08923853375, Real Time Speedup: 1356.0498855754506 (benchmark.py:82)
[18912 ms][__main__] - CRITICAL : i 1751, Current time: 3.3053858280181885, FPS: 135613.57584353746, Real Time Speedup: 1356.1343889799011 (benchmark.py:82)
[19007 ms][__main__] - CRITICAL : i 1801, Current time: 3.3998489379882812, FPS: 135610.4838114601, Real Time Speedup: 1356.1037920336394 (benchmark.py:82)
[19092 ms][__main__] - CRITICAL : i 1851, Current time: 3.4854540824890137, FPS: 135952.22983119934, Real Time Speedup: 1359.5212753517867 (benchmark.py:82)
[19178 ms][__main__] - CRITICAL : i 1901, Current time: 3.571307420730591, FPS: 136268.1296734238, Real Time Speedup: 1362.6803870172316 (benchmark.py:82)
[19274 ms][__main__] - CRITICAL : i 1951, Current time: 3.6666197776794434, FPS: 136216.78436594867, Real Time Speedup: 1362.166869351006 (benchmark.py:82)
[19372 ms][__main__] - CRITICAL : i 2001, Current time: 3.7650179862976074, FPS: 136056.5137650559, Real Time Speedup: 1360.5641899217087 (benchmark.py:82)
[19463 ms][__main__] - CRITICAL : i 2051, Current time: 3.8565802574157715, FPS: 136145.2520332662, Real Time Speedup: 1361.4515945013602 (benchmark.py:82)
[19558 ms][__main__] - CRITICAL : i 2101, Current time: 3.9506523609161377, FPS: 136143.3529325359, Real Time Speedup: 1361.4322147468583 (benchmark.py:82)
[19652 ms][__main__] - CRITICAL : i 2151, Current time: 4.04487681388855, FPS: 136136.46258154875, Real Time Speedup: 1361.3635024105713 (benchmark.py:82)
[19751 ms][__main__] - CRITICAL : i 2201, Current time: 4.14459490776062, FPS: 135949.4309744662, Real Time Speedup: 1359.4934494895024 (benchmark.py:82)
[19850 ms][__main__] - CRITICAL : i 2251, Current time: 4.2431676387786865, FPS: 135807.77551116157, Real Time Speedup: 1358.07676309958 (benchmark.py:82)
[19946 ms][__main__] - CRITICAL : i 2301, Current time: 4.33960485458374, FPS: 135739.3910551921, Real Time Speedup: 1357.3931647988236 (benchmark.py:82)
[20041 ms][__main__] - CRITICAL : i 2351, Current time: 4.434360504150391, FPS: 135725.37072930488, Real Time Speedup: 1357.2528316027472 (benchmark.py:82)
[20134 ms][__main__] - CRITICAL : i 2401, Current time: 4.5266382694244385, FPS: 135786.2400454395, Real Time Speedup: 1357.8617567865724 (benchmark.py:82)
[20242 ms][__main__] - CRITICAL : i 2451, Current time: 4.635202169418335, FPS: 135367.38425073985, Real Time Speedup: 1353.6729373419778 (benchmark.py:82)
[20342 ms][__main__] - CRITICAL : i 2501, Current time: 4.735497713088989, FPS: 135203.34470558667, Real Time Speedup: 1352.0326982766976 (benchmark.py:82)
[20443 ms][__main__] - CRITICAL : i 2551, Current time: 4.836388111114502, FPS: 135029.5661387524, Real Time Speedup: 1350.2951288659197 (benchmark.py:82)
[20540 ms][__main__] - CRITICAL : i 2601, Current time: 4.933098316192627, FPS: 134977.05828105845, Real Time Speedup: 1349.7696695232448 (benchmark.py:82)
[20640 ms][__main__] - CRITICAL : i 2651, Current time: 5.033231973648071, FPS: 134834.88624583438, Real Time Speedup: 1348.3482876311934 (benchmark.py:82)
[20727 ms][__main__] - CRITICAL : i 2701, Current time: 5.120023488998413, FPS: 135049.21692868267, Real Time Speedup: 1350.4914775320779 (benchmark.py:82)
[20821 ms][__main__] - CRITICAL : i 2751, Current time: 5.213642358779907, FPS: 135079.3004574296, Real Time Speedup: 1350.7923250897056 (benchmark.py:82)
[20912 ms][__main__] - CRITICAL : i 2801, Current time: 5.305482625961304, FPS: 135153.62827426317, Real Time Speedup: 1351.535736123569 (benchmark.py:82)
[21007 ms][__main__] - CRITICAL : i 2851, Current time: 5.399950265884399, FPS: 135159.6320563485, Real Time Speedup: 1351.5957238075832 (benchmark.py:82)
[21105 ms][__main__] - CRITICAL : i 2901, Current time: 5.498025894165039, FPS: 135076.6883907877, Real Time Speedup: 1350.7660638566501 (benchmark.py:82)
[21200 ms][__main__] - CRITICAL : i 2951, Current time: 5.5930492877960205, FPS: 135070.34525094595, Real Time Speedup: 1350.7026464283992 (benchmark.py:82)
[21291 ms][__main__] - CRITICAL : i 3001, Current time: 5.684427738189697, FPS: 135150.86455697386, Real Time Speedup: 1351.5080787152253 (benchmark.py:82)
[21386 ms][__main__] - CRITICAL : i 3051, Current time: 5.779244899749756, FPS: 135148.28997812248, Real Time Speedup: 1351.4821192202808 (benchmark.py:82)
[21480 ms][__main__] - CRITICAL : i 3101, Current time: 5.873286485671997, FPS: 135163.71600868428, Real Time Speedup: 1351.6365016719517 (benchmark.py:82)
[21572 ms][__main__] - CRITICAL : i 3151, Current time: 5.964873790740967, FPS: 135234.26500030077, Real Time Speedup: 1352.3419473056128 (benchmark.py:82)
[21664 ms][__main__] - CRITICAL : i 3201, Current time: 6.0575244426727295, FPS: 135278.9184253437, Real Time Speedup: 1352.7886518087107 (benchmark.py:82)
[21758 ms][__main__] - CRITICAL : i 3251, Current time: 6.151012420654297, FPS: 135303.8000746999, Real Time Speedup: 1353.0374762983383 (benchmark.py:82)
[21850 ms][__main__] - CRITICAL : i 3301, Current time: 6.242828607559204, FPS: 135364.1353093789, Real Time Speedup: 1353.6407844316225 (benchmark.py:82)
[21944 ms][__main__] - CRITICAL : i 3351, Current time: 6.337177038192749, FPS: 135368.64934444317, Real Time Speedup: 1353.6858313727178 (benchmark.py:82)
[22048 ms][__main__] - CRITICAL : i 3401, Current time: 6.440912246704102, FPS: 135175.7728760208, Real Time Speedup: 1351.7572283904424 (benchmark.py:82)
[22146 ms][__main__] - CRITICAL : i 3451, Current time: 6.538970947265625, FPS: 135106.13106897022, Real Time Speedup: 1351.0605717715976 (benchmark.py:82)
[22241 ms][__main__] - CRITICAL : i 3501, Current time: 6.634285926818848, FPS: 135094.45765206826, Real Time Speedup: 1350.943945379916 (benchmark.py:82)
[22337 ms][__main__] - CRITICAL : i 3551, Current time: 6.729606866836548, FPS: 135082.95697370535, Real Time Speedup: 1350.8290433040709 (benchmark.py:82)
[22432 ms][__main__] - CRITICAL : i 3601, Current time: 6.824978828430176, FPS: 135070.79138259275, Real Time Speedup: 1350.7074419805053 (benchmark.py:82)
[22526 ms][__main__] - CRITICAL : i 3651, Current time: 6.919039726257324, FPS: 135084.52499362436, Real Time Speedup: 1350.844784458042 (benchmark.py:82)
[22622 ms][__main__] - CRITICAL : i 3701, Current time: 7.01546835899353, FPS: 135052.28984678217, Real Time Speedup: 1350.5222559091658 (benchmark.py:82)
[22716 ms][__main__] - CRITICAL : i 3751, Current time: 7.1092917919158936, FPS: 135070.46609421095, Real Time Speedup: 1350.7041626702185 (benchmark.py:82)
[22807 ms][__main__] - CRITICAL : i 3801, Current time: 7.200483798980713, FPS: 135137.4743257708, Real Time Speedup: 1351.3742957982292 (benchmark.py:82)
[22902 ms][__main__] - CRITICAL : i 3851, Current time: 7.295335531234741, FPS: 135135.01260484126, Real Time Speedup: 1351.3496402513176 (benchmark.py:82)
[22997 ms][__main__] - CRITICAL : i 3901, Current time: 7.3896331787109375, FPS: 135142.73341525783, Real Time Speedup: 1351.4268981295395 (benchmark.py:82)
[23093 ms][__main__] - CRITICAL : i 3951, Current time: 7.486014127731323, FPS: 135112.64838774566, Real Time Speedup: 1351.125881438627 (benchmark.py:82)
[23196 ms][__main__] - CRITICAL : i 4001, Current time: 7.588669300079346, FPS: 134971.6387214475, Real Time Speedup: 1349.714987852367 (benchmark.py:82)
[23293 ms][__main__] - CRITICAL : i 4051, Current time: 7.686261177062988, FPS: 134923.2230818868, Real Time Speedup: 1349.2316867492452 (benchmark.py:82)
[23390 ms][__main__] - CRITICAL : i 4101, Current time: 7.7835447788238525, FPS: 134881.40056804128, Real Time Speedup: 1348.8135512085155 (benchmark.py:82)
[23496 ms][__main__] - CRITICAL : i 4151, Current time: 7.889086723327637, FPS: 134699.3975996798, Real Time Speedup: 1346.9935689179165 (benchmark.py:82)
[23586 ms][__main__] - CRITICAL : i 4201, Current time: 7.97902250289917, FPS: 134785.35610570162, Real Time Speedup: 1347.8531180348955 (benchmark.py:82)
[23687 ms][__main__] - CRITICAL : i 4251, Current time: 8.08045506477356, FPS: 134677.4720922764, Real Time Speedup: 1346.7743235493488 (benchmark.py:82)
[23788 ms][__main__] - CRITICAL : i 4301, Current time: 8.18093490600586, FPS: 134587.95831977812, Real Time Speedup: 1345.8791517426448 (benchmark.py:82)
[23900 ms][__main__] - CRITICAL : i 4351, Current time: 8.293581008911133, FPS: 134303.29966685854, Real Time Speedup: 1343.0325719737687 (benchmark.py:82)
[24000 ms][__main__] - CRITICAL : i 4401, Current time: 8.393552780151367, FPS: 134228.65447477065, Real Time Speedup: 1342.2861634720198 (benchmark.py:82)
[24097 ms][__main__] - CRITICAL : i 4451, Current time: 8.49046802520752, FPS: 134204.0807023078, Real Time Speedup: 1342.0404678539005 (benchmark.py:82)
[24193 ms][__main__] - CRITICAL : i 4501, Current time: 8.58583688735962, FPS: 134204.19874979276, Real Time Speedup: 1342.041614828983 (benchmark.py:82)
[24299 ms][__main__] - CRITICAL : i 4551, Current time: 8.692224979400635, FPS: 134034.1709860096, Real Time Speedup: 1340.3413054547261 (benchmark.py:82)
[24396 ms][__main__] - CRITICAL : i 4601, Current time: 8.788776397705078, FPS: 134018.1026328773, Real Time Speedup: 1340.1806627699327 (benchmark.py:82)
[24489 ms][__main__] - CRITICAL : i 4651, Current time: 8.881841897964478, FPS: 134054.98349184825, Real Time Speedup: 1340.5494390852302 (benchmark.py:82)
[24589 ms][__main__] - CRITICAL : i 4701, Current time: 8.982481718063354, FPS: 133978.04381610436, Real Time Speedup: 1339.7800469872946 (benchmark.py:82)
[24692 ms][__main__] - CRITICAL : i 4751, Current time: 9.08526062965393, FPS: 133871.24001079818, Real Time Speedup: 1338.7118380133052 (benchmark.py:82)
[24792 ms][__main__] - CRITICAL : i 4801, Current time: 9.18464994430542, FPS: 133816.22705775077, Real Time Speedup: 1338.1618537397064 (benchmark.py:82)
[24885 ms][__main__] - CRITICAL : i 4851, Current time: 9.278490781784058, FPS: 133842.37064452, Real Time Speedup: 1338.423396918224 (benchmark.py:82)
[24983 ms][__main__] - CRITICAL : i 4901, Current time: 9.37600564956665, FPS: 133815.5475697289, Real Time Speedup: 1338.1551354235226 (benchmark.py:82)
[25082 ms][__main__] - CRITICAL : i 4951, Current time: 9.475021839141846, FPS: 133768.05416290712, Real Time Speedup: 1337.6801377111356 (benchmark.py:82)
[25188 ms][__main__] - CRITICAL : i 5001, Current time: 9.580869913101196, FPS: 133626.20625419324, Real Time Speedup: 1336.261696762637 (benchmark.py:82)
[25290 ms][__main__] - CRITICAL : i 5051, Current time: 9.682663679122925, FPS: 133543.35191912844, Real Time Speedup: 1335.4331574818182 (benchmark.py:82)
[25384 ms][__main__] - CRITICAL : i 5101, Current time: 9.77692461013794, FPS: 133565.03070307913, Real Time Speedup: 1335.6499813214327 (benchmark.py:82)
[25490 ms][__main__] - CRITICAL : i 5151, Current time: 9.882770776748657, FPS: 133429.6775850087, Real Time Speedup: 1334.2962608192076 (benchmark.py:82)
[25600 ms][__main__] - CRITICAL : i 5201, Current time: 9.99274206161499, FPS: 133242.21107356952, Real Time Speedup: 1332.4216338788274 (benchmark.py:82)
[25709 ms][__main__] - CRITICAL : i 5251, Current time: 10.10202407836914, FPS: 133067.904481241, Real Time Speedup: 1330.6787307581817 (benchmark.py:82)
[25814 ms][__main__] - CRITICAL : i 5301, Current time: 10.20735502243042, FPS: 132948.78625864102, Real Time Speedup: 1329.4875520511084 (benchmark.py:82)
[25917 ms][__main__] - CRITICAL : i 5351, Current time: 10.310097932815552, FPS: 132865.410886727, Real Time Speedup: 1328.6538323441375 (benchmark.py:82)
[26014 ms][__main__] - CRITICAL : i 5401, Current time: 10.407026290893555, FPS: 132857.84457105538, Real Time Speedup: 1328.5780804679046 (benchmark.py:82)
[26114 ms][__main__] - CRITICAL : i 5451, Current time: 10.507575988769531, FPS: 132804.68650107895, Real Time Speedup: 1328.0465335414776 (benchmark.py:82)
[26211 ms][__main__] - CRITICAL : i 5501, Current time: 10.603981018066406, FPS: 132804.4028313481, Real Time Speedup: 1328.0437297179287 (benchmark.py:82)
[26311 ms][__main__] - CRITICAL : i 5551, Current time: 10.70390272140503, FPS: 132760.48082436871, Real Time Speedup: 1327.6045125333956 (benchmark.py:82)
[26413 ms][__main__] - CRITICAL : i 5601, Current time: 10.80563235282898, FPS: 132695.14537980774, Real Time Speedup: 1326.9511317377676 (benchmark.py:82)
[26513 ms][__main__] - CRITICAL : i 5651, Current time: 10.906472206115723, FPS: 132641.90125395986, Real Time Speedup: 1326.418693584991 (benchmark.py:82)
[26618 ms][__main__] - CRITICAL : i 5701, Current time: 11.01066255569458, FPS: 132549.26901691259, Real Time Speedup: 1325.4924031547375 (benchmark.py:82)
[26721 ms][__main__] - CRITICAL : i 5751, Current time: 11.114139318466187, FPS: 132466.88080493393, Real Time Speedup: 1324.6685523004194 (benchmark.py:82)
[26825 ms][__main__] - CRITICAL : i 5801, Current time: 11.218370199203491, FPS: 132377.08521637236, Real Time Speedup: 1323.7705145624873 (benchmark.py:82)
[26930 ms][__main__] - CRITICAL : i 5851, Current time: 11.323277473449707, FPS: 132281.06849750486, Real Time Speedup: 1322.8104064493136 (benchmark.py:82)
[27034 ms][__main__] - CRITICAL : i 5901, Current time: 11.426788568496704, FPS: 132202.9657792322, Real Time Speedup: 1322.029381952591 (benchmark.py:82)
[27152 ms][__main__] - CRITICAL : i 5951, Current time: 11.544783353805542, FPS: 131960.49342385324, Real Time Speedup: 1319.6046344668343 (benchmark.py:82)
[27257 ms][__main__] - CRITICAL : i 6001, Current time: 11.65030574798584, FPS: 131863.93155208667, Real Time Speedup: 1318.6389107402856 (benchmark.py:82)
[27373 ms][__main__] - CRITICAL : i 6051, Current time: 11.766180992126465, FPS: 131653.19939113647, Real Time Speedup: 1316.5317804959773 (benchmark.py:82)
[27512 ms][__main__] - CRITICAL : i 6101, Current time: 11.904765129089355, FPS: 131195.80803443835, Real Time Speedup: 1311.9577913219111 (benchmark.py:82)
[27619 ms][__main__] - CRITICAL : i 6151, Current time: 12.012109279632568, FPS: 131088.99326122622, Real Time Speedup: 1310.8896464057905 (benchmark.py:82)
[27730 ms][__main__] - CRITICAL : i 6201, Current time: 12.122636318206787, FPS: 130949.67981331408, Real Time Speedup: 1309.4965148375131 (benchmark.py:82)
[27836 ms][__main__] - CRITICAL : i 6251, Current time: 12.228993892669678, FPS: 130857.46896881518, Real Time Speedup: 1308.574434566347 (benchmark.py:82)
[27940 ms][__main__] - CRITICAL : i 6301, Current time: 12.333528757095337, FPS: 130786.18120382412, Real Time Speedup: 1307.8615086522566 (benchmark.py:82)
[28041 ms][__main__] - CRITICAL : i 6351, Current time: 12.434564352035522, FPS: 130752.87962865319, Real Time Speedup: 1307.5284453016088 (benchmark.py:82)
[28140 ms][__main__] - CRITICAL : i 6401, Current time: 12.532732009887695, FPS: 130750.03823611538, Real Time Speedup: 1307.5000590064396 (benchmark.py:82)
[28242 ms][__main__] - CRITICAL : i 6451, Current time: 12.635496139526367, FPS: 130699.68097942806, Real Time Speedup: 1306.9965631778196 (benchmark.py:82)
[28334 ms][__main__] - CRITICAL : i 6501, Current time: 12.72698712348938, FPS: 130765.85196147497, Real Time Speedup: 1307.658299143941 (benchmark.py:82)
[28436 ms][__main__] - CRITICAL : i 6551, Current time: 12.829030990600586, FPS: 130723.44744684627, Real Time Speedup: 1307.2341829401998 (benchmark.py:82)
[28532 ms][__main__] - CRITICAL : i 6601, Current time: 12.924835443496704, FPS: 130744.81995779727, Real Time Speedup: 1307.4479101633103 (benchmark.py:82)
[28637 ms][__main__] - CRITICAL : i 6651, Current time: 13.029670000076294, FPS: 130675.25657580563, Real Time Speedup: 1306.7523266469077 (benchmark.py:82)
[28738 ms][__main__] - CRITICAL : i 6701, Current time: 13.131103992462158, FPS: 130640.584197461, Real Time Speedup: 1306.405509893123 (benchmark.py:82)
[28853 ms][__main__] - CRITICAL : i 6751, Current time: 13.245914459228516, FPS: 130474.58963577427, Real Time Speedup: 1304.7456380267574 (benchmark.py:82)
[28959 ms][__main__] - CRITICAL : i 6801, Current time: 13.352574586868286, FPS: 130390.97061822432, Real Time Speedup: 1303.909426796948 (benchmark.py:82)
[29058 ms][__main__] - CRITICAL : i 6851, Current time: 13.451145887374878, FPS: 130387.06323145927, Real Time Speedup: 1303.8704243173345 (benchmark.py:82)
[29166 ms][__main__] - CRITICAL : i 6901, Current time: 13.559269905090332, FPS: 130291.32363092556, Real Time Speedup: 1302.9129155737617 (benchmark.py:82)
[29277 ms][__main__] - CRITICAL : i 6951, Current time: 13.669979572296143, FPS: 130172.48370160525, Real Time Speedup: 1301.7245418716998 (benchmark.py:82)
[29385 ms][__main__] - CRITICAL : i 7001, Current time: 13.778279304504395, FPS: 130078.32125877158, Real Time Speedup: 1300.783010009687 (benchmark.py:82)
[29486 ms][__main__] - CRITICAL : i 7051, Current time: 13.878760814666748, FPS: 130058.82678770612, Real Time Speedup: 1300.5880667958365 (benchmark.py:82)
[29591 ms][__main__] - CRITICAL : i 7101, Current time: 13.983741283416748, FPS: 129997.77744003474, Real Time Speedup: 1299.9774862655383 (benchmark.py:82)
[29699 ms][__main__] - CRITICAL : i 7151, Current time: 14.091897010803223, FPS: 129908.35749862881, Real Time Speedup: 1299.0833112385808 (benchmark.py:82)
[29809 ms][__main__] - CRITICAL : i 7201, Current time: 14.202545404434204, FPS: 129797.5238840582, Real Time Speedup: 1297.9749773709277 (benchmark.py:82)
[29916 ms][__main__] - CRITICAL : i 7251, Current time: 14.308645963668823, FPS: 129729.62161929427, Real Time Speedup: 1297.2960000303926 (benchmark.py:82)
[30020 ms][__main__] - CRITICAL : i 7301, Current time: 14.413513660430908, FPS: 129673.80538041014, Real Time Speedup: 1296.7378822061012 (benchmark.py:82)
[30127 ms][__main__] - CRITICAL : i 7351, Current time: 14.520431518554688, FPS: 129600.50505491278, Real Time Speedup: 1296.0048377513817 (benchmark.py:82)
[30228 ms][__main__] - CRITICAL : i 7401, Current time: 14.621189832687378, FPS: 129582.84120456636, Real Time Speedup: 1295.8281796129093 (benchmark.py:82)
[30331 ms][__main__] - CRITICAL : i 7451, Current time: 14.724605321884155, FPS: 129542.03381533589, Real Time Speedup: 1295.4201703514411 (benchmark.py:82)
[30432 ms][__main__] - CRITICAL : i 7501, Current time: 14.825097560882568, FPS: 129527.32903338029, Real Time Speedup: 1295.2730820268746 (benchmark.py:82)
[30544 ms][__main__] - CRITICAL : i 7551, Current time: 14.937409400939941, FPS: 129410.34408819668, Real Time Speedup: 1294.1032343279837 (benchmark.py:82)
[30654 ms][__main__] - CRITICAL : i 7601, Current time: 15.046620845794678, FPS: 129321.75087384951, Real Time Speedup: 1293.2172833326579 (benchmark.py:82)
[30753 ms][__main__] - CRITICAL : i 7651, Current time: 15.146230936050415, FPS: 129316.35544455204, Real Time Speedup: 1293.1633508872405 (benchmark.py:82)
[30863 ms][__main__] - CRITICAL : i 7701, Current time: 15.255767583847046, FPS: 129226.88291707658, Real Time Speedup: 1292.2686272138749 (benchmark.py:82)
[30968 ms][__main__] - CRITICAL : i 7751, Current time: 15.360968112945557, FPS: 129175.13952692683, Real Time Speedup: 1291.7512148251542 (benchmark.py:82)
[31075 ms][__main__] - CRITICAL : i 7801, Current time: 15.468131303787231, FPS: 129107.71426285383, Real Time Speedup: 1291.0769038279443 (benchmark.py:82)
[31174 ms][__main__] - CRITICAL : i 7851, Current time: 15.56710410118103, FPS: 129109.13159893091, Real Time Speedup: 1291.0911182518241 (benchmark.py:82)
[31273 ms][__main__] - CRITICAL : i 7901, Current time: 15.666349649429321, FPS: 129108.26160227158, Real Time Speedup: 1291.082360594187 (benchmark.py:82)
[31386 ms][__main__] - CRITICAL : i 7951, Current time: 15.778749465942383, FPS: 128999.7891479645, Real Time Speedup: 1289.9976770676933 (benchmark.py:82)
[31485 ms][__main__] - CRITICAL : i 8001, Current time: 15.877814769744873, FPS: 129001.08247320558, Real Time Speedup: 1290.0106116557383 (benchmark.py:82)
[31591 ms][__main__] - CRITICAL : i 8051, Current time: 15.98365306854248, FPS: 128947.70335338937, Real Time Speedup: 1289.4768604247622 (benchmark.py:82)
[31699 ms][__main__] - CRITICAL : i 8101, Current time: 16.092392444610596, FPS: 128871.7821450916, Real Time Speedup: 1288.717630519628 (benchmark.py:82)
[31814 ms][__main__] - CRITICAL : i 8151, Current time: 16.206953525543213, FPS: 128750.60924980447, Real Time Speedup: 1287.5058462736424 (benchmark.py:82)
[31924 ms][__main__] - CRITICAL : i 8201, Current time: 16.31701922416687, FPS: 128666.58688646399, Real Time Speedup: 1286.6656056604331 (benchmark.py:82)
[32039 ms][__main__] - CRITICAL : i 8251, Current time: 16.432498693466187, FPS: 128541.33538521011, Real Time Speedup: 1285.4131487019822 (benchmark.py:82)
[32188 ms][__main__] - CRITICAL : i 8301, Current time: 16.58122682571411, FPS: 128160.31766244535, Real Time Speedup: 1281.6029554891725 (benchmark.py:82)
[32289 ms][__main__] - CRITICAL : i 8351, Current time: 16.68181538581848, FPS: 128154.82944384034, Real Time Speedup: 1281.5480929619873 (benchmark.py:82)
[32391 ms][__main__] - CRITICAL : i 8401, Current time: 16.78411340713501, FPS: 128136.36957039573, Real Time Speedup: 1281.3634954843558 (benchmark.py:82)
[32503 ms][__main__] - CRITICAL : i 8451, Current time: 16.89659833908081, FPS: 128040.8678298121, Real Time Speedup: 1280.4084072911803 (benchmark.py:82)
[32615 ms][__main__] - CRITICAL : i 8501, Current time: 17.00805163383484, FPS: 127954.4222480583, Real Time Speedup: 1279.544043114364 (benchmark.py:82)
[32728 ms][__main__] - CRITICAL : i 8551, Current time: 17.120840787887573, FPS: 127859.09848148661, Real Time Speedup: 1278.590788957857 (benchmark.py:82)
[32843 ms][__main__] - CRITICAL : i 8601, Current time: 17.236361026763916, FPS: 127744.77645711915, Real Time Speedup: 1277.4475171907075 (benchmark.py:82)
[32950 ms][__main__] - CRITICAL : i 8651, Current time: 17.342790842056274, FPS: 127698.89394266062, Real Time Speedup: 1276.9887638736975 (benchmark.py:82)
[33055 ms][__main__] - CRITICAL : i 8701, Current time: 17.448054790496826, FPS: 127662.08864381808, Real Time Speedup: 1276.6207119946803 (benchmark.py:82)
[33164 ms][__main__] - CRITICAL : i 8751, Current time: 17.556930541992188, FPS: 127599.46275613757, Real Time Speedup: 1275.9943156634138 (benchmark.py:82)
[33278 ms][__main__] - CRITICAL : i 8801, Current time: 17.670958042144775, FPS: 127500.45951459113, Real Time Speedup: 1275.0043715134289 (benchmark.py:82)
[33397 ms][__main__] - CRITICAL : i 8851, Current time: 17.79040813446045, FPS: 127363.87067271137, Real Time Speedup: 1273.6385189715097 (benchmark.py:82)
[33503 ms][__main__] - CRITICAL : i 8901, Current time: 17.89602541923523, FPS: 127327.44263698043, Real Time Speedup: 1273.2742228125715 (benchmark.py:82)
[33617 ms][__main__] - CRITICAL : i 8951, Current time: 18.010308027267456, FPS: 127230.20660421709, Real Time Speedup: 1272.301863931048 (benchmark.py:82)
[33729 ms][__main__] - CRITICAL : i 9001, Current time: 18.1216139793396, FPS: 127155.07816929267, Real Time Speedup: 1271.5505809418125 (benchmark.py:82)
[33843 ms][__main__] - CRITICAL : i 9051, Current time: 18.236163854599, FPS: 127058.26300609253, Real Time Speedup: 1270.5824307227078 (benchmark.py:82)
[33954 ms][__main__] - CRITICAL : i 9101, Current time: 18.346614599227905, FPS: 126991.01021419934, Real Time Speedup: 1269.9099041087275 (benchmark.py:82)
[34065 ms][__main__] - CRITICAL : i 9151, Current time: 18.458099126815796, FPS: 126917.47171132095, Real Time Speedup: 1269.1745203900107 (benchmark.py:82)
[34176 ms][__main__] - CRITICAL : i 9201, Current time: 18.56931710243225, FPS: 126846.61254685812, Real Time Speedup: 1268.4658323149677 (benchmark.py:82)
[34290 ms][__main__] - CRITICAL : i 9251, Current time: 18.68324303627014, FPS: 126758.25282925437, Real Time Speedup: 1267.5823180080652 (benchmark.py:82)
[34400 ms][__main__] - CRITICAL : i 9301, Current time: 18.793136835098267, FPS: 126698.11935480339, Real Time Speedup: 1266.9810006659 (benchmark.py:82)
[34517 ms][__main__] - CRITICAL : i 9351, Current time: 18.909693717956543, FPS: 126594.08140133147, Real Time Speedup: 1265.9406065161543 (benchmark.py:82)
[34633 ms][__main__] - CRITICAL : i 9401, Current time: 19.02606725692749, FPS: 126492.51803181053, Real Time Speedup: 1264.924974255572 (benchmark.py:82)
[34741 ms][__main__] - CRITICAL : i 9451, Current time: 19.13444185256958, FPS: 126445.0405774843, Real Time Speedup: 1264.4502167115802 (benchmark.py:82)
[34855 ms][__main__] - CRITICAL : i 9501, Current time: 19.248107433319092, FPS: 126363.34847446758, Real Time Speedup: 1263.6333125714211 (benchmark.py:82)
[34963 ms][__main__] - CRITICAL : i 9551, Current time: 19.35637664794922, FPS: 126317.80942140693, Real Time Speedup: 1263.1778452708204 (benchmark.py:82)
[35087 ms][__main__] - CRITICAL : i 9601, Current time: 19.480154275894165, FPS: 126172.26810139618, Real Time Speedup: 1261.7225265911602 (benchmark.py:82)
[35202 ms][__main__] - CRITICAL : i 9651, Current time: 19.5946044921875, FPS: 126088.54438747269, Real Time Speedup: 1260.8852751138772 (benchmark.py:82)
[35308 ms][__main__] - CRITICAL : i 9701, Current time: 19.701322555541992, FPS: 126055.25246048621, Real Time Speedup: 1260.55234154771 (benchmark.py:82)
[35430 ms][__main__] - CRITICAL : i 9751, Current time: 19.82329535484314, FPS: 125925.3436305952, Real Time Speedup: 1259.2532545626366 (benchmark.py:82)
[35567 ms][__main__] - CRITICAL : i 9801, Current time: 19.96056866645813, FPS: 125700.57461784135, Real Time Speedup: 1257.0055059501144 (benchmark.py:82)
[35678 ms][__main__] - CRITICAL : i 9851, Current time: 20.070876359939575, FPS: 125647.48773016775, Real Time Speedup: 1256.474668345414 (benchmark.py:82)


## Benchmark.py - physics only (old code)

docker compose -f docker/docker-compose.yaml run --rm \
  aerialgym python aerial_gym/examples/benchmark.py
WARN[0000] Found orphan containers ([mini_aerial_gym_newton_x86_64]) for this project. If you removed or renamed this service in your compose file, you can run this command with the --remove-orphans flag to clean it up. 
Container docker-aerialgym-run-266bc5548ad0 Creating 
Container docker-aerialgym-run-266bc5548ad0 Created 
/bin/bash: /opt/conda/envs/aerialgym/lib/libtinfo.so.6: no version information available (required by /bin/bash)
Importing module 'gym_38' (/opt/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so)
Setting GYM_USD_PLUG_INFO_PATH to /opt/isaacgym/python/isaacgym/_bindings/linux-x86_64/usd/plugInfo.json
PyTorch version 2.4.1+cu121
Device count 1
/opt/isaacgym/python/isaacgym/_bindings/src/gymtorch
Using /root/.cache/torch_extensions/py38_cu121 as PyTorch extensions root...
Creating extension directory /root/.cache/torch_extensions/py38_cu121/gymtorch...
Emitting ninja build file /root/.cache/torch_extensions/py38_cu121/gymtorch/build.ninja...
Building extension module gymtorch...
Allowing ninja to set a default number of workers... (overridable by setting the environment variable MAX_JOBS=N)
[1/2] c++ -MMD -MF gymtorch.o.d -DTORCH_EXTENSION_NAME=gymtorch -DTORCH_API_INCLUDE_EXTENSION_H -DPYBIND11_COMPILER_TYPE=\"_gcc\" -DPYBIND11_STDLIB=\"_libstdcpp\" -DPYBIND11_BUILD_ABI=\"_cxxabi1011\" -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/torch/csrc/api/include -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/TH -isystem /opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/include/THC -isystem /opt/conda/envs/aerialgym/include/python3.8 -D_GLIBCXX_USE_CXX11_ABI=0 -fPIC -std=c++17 -DTORCH_MAJOR=2 -DTORCH_MINOR=4 -c /opt/isaacgym/python/isaacgym/_bindings/src/gymtorch/gymtorch.cpp -o gymtorch.o 
[2/2] c++ gymtorch.o -shared -L/opt/conda/envs/aerialgym/lib/python3.8/site-packages/torch/lib -lc10 -ltorch_cpu -ltorch -ltorch_python -o gymtorch.so
Loading extension module gymtorch...
Warp UserWarning: Python 3.9 or newer is recommended for running Warp, detected sys.version_info(major=3, minor=8, micro=20, releaselevel='final', serial=0)
Warp 1.10.1 initialized:
   CUDA Toolkit 12.8, Driver 13.0
   Devices:
     "cpu"      : "x86_64"
     "cuda:0"   : "NVIDIA GeForce RTX 4080 Laptop GPU" (12 GiB, sm_89, mempool enabled)
   Kernel cache:
     /root/.cache/warp/1.10.1
Gym has been unmaintained since 2022 and does not support NumPy 2.0 amongst other critical functionality.
Please upgrade to Gymnasium, the maintained drop-in replacement of Gym, or contact the authors of your software and request that they upgrade.
See the migration guide at https://gymnasium.farama.org/introduction/migration_guide/ for additional information.
[13020 ms][__main__] - WARNING : This script provides an example of a rendering benchmark for the environment. The rendering benchmark will measure the FPS and the real-time speedup of the environment. (benchmark.py:24)
[13020 ms][__main__] - WARNING : 


The rendering benchmark will run by default. Please set rendering_benchmark = False to run the physics benchmark. 


 (benchmark.py:27)
[13020 ms][env_manager] - INFO : Populating environments. (env_manager.py:73)
[13021 ms][env_manager] - INFO : Creating simulation instance. (env_manager.py:87)
[13021 ms][env_manager] - INFO : Instantiating IGE object. (env_manager.py:88)
[13021 ms][IsaacGymEnvManager] - INFO : Creating Isaac Gym Environment (IGE_env_manager.py:41)
[13021 ms][IsaacGymEnvManager] - INFO : Acquiring gym object (IGE_env_manager.py:73)
[13021 ms][IsaacGymEnvManager] - INFO : Acquired gym object (IGE_env_manager.py:75)
[isaacgym:gymutil.py] Unknown args:  []
[13021 ms][IsaacGymEnvManager] - INFO : Fixing devices (IGE_env_manager.py:89)
[13021 ms][IsaacGymEnvManager] - INFO : Using GPU pipeline for simulation. (IGE_env_manager.py:102)
[13021 ms][IsaacGymEnvManager] - INFO : Sim Device type: cuda, Sim Device ID: 0 (IGE_env_manager.py:105)
[13021 ms][IsaacGymEnvManager] - CRITICAL : 
 Setting graphics device to -1.
 This is done because the simulation is run in headless mode and no Isaac Gym cameras are used.
 No need to worry. The simulation and warp rendering will work as expected. (IGE_env_manager.py:112)
[13021 ms][IsaacGymEnvManager] - INFO : Graphics Device ID: -1 (IGE_env_manager.py:119)
[13021 ms][IsaacGymEnvManager] - INFO : Creating Isaac Gym Simulation Object (IGE_env_manager.py:120)
[13022 ms][IsaacGymEnvManager] - WARNING : If you have set the CUDA_VISIBLE_DEVICES environment variable, please ensure that you set it
to a particular one that works for your system to use the viewer or Isaac Gym cameras.
If you want to run parallel simulations on multiple GPUs with camera sensors,
please disable Isaac Gym and use warp (by setting use_warp=True), set the viewer to headless. (IGE_env_manager.py:127)
[13022 ms][IsaacGymEnvManager] - WARNING : If you see a segfault in the next lines, it is because of the discrepancy between the CUDA device and the graphics device.
Please ensure that the CUDA device and the graphics device are the same. (IGE_env_manager.py:132)
Not connected to PVD
+++ Using GPU PhysX
Physics Engine: PhysX
Physics Device: cuda:0
GPU Pipeline: enabled
[14045 ms][IsaacGymEnvManager] - INFO : Created Isaac Gym Simulation Object (IGE_env_manager.py:136)
[14045 ms][IsaacGymEnvManager] - INFO : Created Isaac Gym Environment (IGE_env_manager.py:43)
[14190 ms][env_manager] - INFO : IGE object instantiated. (env_manager.py:109)
[14190 ms][env_manager] - INFO : Creating warp environment. (env_manager.py:112)
[14190 ms][env_manager] - INFO : Warp environment created. (env_manager.py:114)
[14191 ms][env_manager] - INFO : Creating robot manager. (env_manager.py:118)
[14191 ms][BaseRobot] - INFO : [DONE] Initializing controller (base_robot.py:26)
[14191 ms][BaseRobot] - INFO : Initializing controller no_control (base_robot.py:29)
[14191 ms][base_multirotor] - WARNING : Creating 256 multirotors. (base_multirotor.py:32)
[14191 ms][env_manager] - INFO : [DONE] Creating robot manager. (env_manager.py:123)
[14191 ms][env_manager] - INFO : [DONE] Creating simulation instance. (env_manager.py:125)
[14191 ms][asset_loader] - INFO : Loading asset: quad.urdf for the first time. Next use of this asset will be via the asset buffer. (asset_loader.py:72)
[14191 ms][env_manager] - INFO : Populating environment 0 (env_manager.py:179)
[14327 ms][robot_manager] - WARNING : 
Robot mass: 0.2500003944120692,
Inertia: tensor([[8.4501e-04, 0.0000e+00, 3.3087e-24],
        [0.0000e+00, 8.4501e-04, 3.3087e-24],
        [3.3087e-24, 3.3087e-24, 1.6900e-03]], device='cuda:0'),
Robot COM: tensor([[0.0000e+00, 0.0000e+00, 1.3411e-14, 1.0000e+00]], device='cuda:0') (robot_manager.py:427)
[14327 ms][robot_manager] - WARNING : Calculated robot mass and inertia for this robot. This code assumes that your robot is the same across environments. (robot_manager.py:430)
[14327 ms][robot_manager] - CRITICAL : If your robot differs across environments you need to perform this computation for each different robot here. (robot_manager.py:433)
[14346 ms][env_manager] - INFO : [DONE] Populating environments. (env_manager.py:75)
[14354 ms][IsaacGymEnvManager] - WARNING : Headless: True (IGE_env_manager.py:424)
[14354 ms][IsaacGymEnvManager] - INFO : Headless mode. Viewer not created. (IGE_env_manager.py:434)
*** Can't create empty tensor
[14355 ms][warp_env_manager] - WARNING : No assets have been added to the environment. Skipping preparation for simulation (warp_env_manager.py:101)
[14357 ms][asset_manager] - WARNING : Number of obstacles to be kept in the environment: 0 (asset_manager.py:32)
WARNING: allocation matrix is not full rank. Rank: 4
/opt/aerialgym/aerial_gym/control/motor_model.py:45: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(self.min_thrust, device=self.device, dtype=torch.float32).expand(
/opt/aerialgym/aerial_gym/control/motor_model.py:48: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).
  torch.tensor(self.max_thrust, device=self.device, dtype=torch.float32).expand(
[14840 ms][control_allocation] - WARNING : Control allocation does not account for actuator limits. This leads to suboptimal allocation (control_allocation.py:48)
[16036 ms][__main__] - CRITICAL : i -99, Current time: 3.0157322883605957, FPS: -8403.914371819887, Real Time Speedup: -84.03907727847336 (benchmark.py:82)
[16460 ms][__main__] - CRITICAL : i -49, Current time: 3.4404966831207275, FPS: -3645.9798978691674, Real Time Speedup: -36.45977876608449 (benchmark.py:82)
[16549 ms][__main__] - CRITICAL : i 1, Current time: 0.0017669200897216797, FPS: 144417.1921990585, Real Time Speedup: 1442.0384421165727 (benchmark.py:82)
[16638 ms][__main__] - CRITICAL : i 51, Current time: 0.09045910835266113, FPS: 144322.80013599238, Real Time Speedup: 1443.1823590770732 (benchmark.py:82)
[16726 ms][__main__] - CRITICAL : i 101, Current time: 0.1783130168914795, FPS: 144999.36386779335, Real Time Speedup: 1449.964558738393 (benchmark.py:82)
[16812 ms][__main__] - CRITICAL : i 151, Current time: 0.26506590843200684, FPS: 145832.80378705633, Real Time Speedup: 1458.312297672511 (benchmark.py:82)
[16897 ms][__main__] - CRITICAL : i 201, Current time: 0.35022974014282227, FPS: 146918.40371193056, Real Time Speedup: 1469.1720356567882 (benchmark.py:82)
[16985 ms][__main__] - CRITICAL : i 251, Current time: 0.4375460147857666, FPS: 146853.4908929423, Real Time Speedup: 1468.523706307462 (benchmark.py:82)
[17074 ms][__main__] - CRITICAL : i 301, Current time: 0.5270082950592041, FPS: 146212.5621304211, Real Time Speedup: 1462.1196681963606 (benchmark.py:82)
[17163 ms][__main__] - CRITICAL : i 351, Current time: 0.6157004833221436, FPS: 145939.73427032144, Real Time Speedup: 1459.391126417732 (benchmark.py:82)
[17256 ms][__main__] - CRITICAL : i 401, Current time: 0.7090365886688232, FPS: 144781.39953428708, Real Time Speedup: 1447.8081533531026 (benchmark.py:82)
[17348 ms][__main__] - CRITICAL : i 451, Current time: 0.801241397857666, FPS: 144095.4983763659, Real Time Speedup: 1440.9502673088377 (benchmark.py:82)
[17441 ms][__main__] - CRITICAL : i 501, Current time: 0.8938112258911133, FPS: 143492.58305060875, Real Time Speedup: 1434.9208547033168 (benchmark.py:82)
[17531 ms][__main__] - CRITICAL : i 551, Current time: 0.9842698574066162, FPS: 143309.77061105886, Real Time Speedup: 1433.0942347551759 (benchmark.py:82)
[17620 ms][__main__] - CRITICAL : i 601, Current time: 1.0725011825561523, FPS: 143454.58620550908, Real Time Speedup: 1434.5423541635637 (benchmark.py:82)
[17715 ms][__main__] - CRITICAL : i 651, Current time: 1.1677963733673096, FPS: 142709.37419591402, Real Time Speedup: 1427.091702469215 (benchmark.py:82)
[17810 ms][__main__] - CRITICAL : i 701, Current time: 1.262650728225708, FPS: 142125.70029684852, Real Time Speedup: 1421.2537825848628 (benchmark.py:82)
[17906 ms][__main__] - CRITICAL : i 751, Current time: 1.358710765838623, FPS: 141498.2957710198, Real Time Speedup: 1414.980226504195 (benchmark.py:82)
[18000 ms][__main__] - CRITICAL : i 801, Current time: 1.4532527923583984, FPS: 141100.934576937, Real Time Speedup: 1411.0070308989675 (benchmark.py:82)
[18091 ms][__main__] - CRITICAL : i 851, Current time: 1.544062614440918, FPS: 141092.3970160044, Real Time Speedup: 1410.9224451406592 (benchmark.py:82)
[18181 ms][__main__] - CRITICAL : i 901, Current time: 1.633643388748169, FPS: 141190.80658971216, Real Time Speedup: 1411.9062113831199 (benchmark.py:82)
[18272 ms][__main__] - CRITICAL : i 951, Current time: 1.7243773937225342, FPS: 141184.51969276252, Real Time Speedup: 1411.843635279151 (benchmark.py:82)
[18377 ms][__main__] - CRITICAL : i 1001, Current time: 1.8301429748535156, FPS: 140019.24205247936, Real Time Speedup: 1400.1902316442317 (benchmark.py:82)
[18474 ms][__main__] - CRITICAL : i 1051, Current time: 1.9271342754364014, FPS: 139614.22978332566, Real Time Speedup: 1396.140743304324 (benchmark.py:82)
[18574 ms][__main__] - CRITICAL : i 1101, Current time: 2.026376485824585, FPS: 139093.25614512322, Real Time Speedup: 1390.9310885724408 (benchmark.py:82)
[18662 ms][__main__] - CRITICAL : i 1151, Current time: 2.115215301513672, FPS: 139302.76295836415, Real Time Speedup: 1393.026216439911 (benchmark.py:82)
[18754 ms][__main__] - CRITICAL : i 1201, Current time: 2.2070162296295166, FPS: 139308.08425652547, Real Time Speedup: 1393.0796386404174 (benchmark.py:82)
[18849 ms][__main__] - CRITICAL : i 1251, Current time: 2.301450252532959, FPS: 139153.72243554104, Real Time Speedup: 1391.5359269533544 (benchmark.py:82)
[18950 ms][__main__] - CRITICAL : i 1301, Current time: 2.4032142162323, FPS: 138587.48129794662, Real Time Speedup: 1385.8733005920728 (benchmark.py:82)
[19045 ms][__main__] - CRITICAL : i 1351, Current time: 2.497390031814575, FPS: 138486.74066665134, Real Time Speedup: 1384.8663489949583 (benchmark.py:82)
[19146 ms][__main__] - CRITICAL : i 1401, Current time: 2.5985841751098633, FPS: 138019.45720106023, Real Time Speedup: 1380.1930524312181 (benchmark.py:82)
[19245 ms][__main__] - CRITICAL : i 1451, Current time: 2.6979658603668213, FPS: 137679.80336559206, Real Time Speedup: 1376.7969386528027 (benchmark.py:82)
[19343 ms][__main__] - CRITICAL : i 1501, Current time: 2.7958552837371826, FPS: 137437.45914866, Real Time Speedup: 1374.3733022816903 (benchmark.py:82)
[19446 ms][__main__] - CRITICAL : i 1551, Current time: 2.899118185043335, FPS: 136957.29330329556, Real Time Speedup: 1369.5715814600453 (benchmark.py:82)
[19542 ms][__main__] - CRITICAL : i 1601, Current time: 2.994718313217163, FPS: 136859.40952256302, Real Time Speedup: 1368.592896691442 (benchmark.py:82)
[19640 ms][__main__] - CRITICAL : i 1651, Current time: 3.0932230949401855, FPS: 136639.1310629717, Real Time Speedup: 1366.3903627669852 (benchmark.py:82)
[19736 ms][__main__] - CRITICAL : i 1701, Current time: 3.1887872219085693, FPS: 136558.3264764377, Real Time Speedup: 1365.5823458513564 (benchmark.py:82)
[19833 ms][__main__] - CRITICAL : i 1751, Current time: 3.2855663299560547, FPS: 136431.70445584622, Real Time Speedup: 1364.3160545377548 (benchmark.py:82)
[19925 ms][__main__] - CRITICAL : i 1801, Current time: 3.377722978591919, FPS: 136498.89279741238, Real Time Speedup: 1364.9879644904324 (benchmark.py:82)
[20022 ms][__main__] - CRITICAL : i 1851, Current time: 3.4749033451080322, FPS: 136365.04463581406, Real Time Speedup: 1363.6496043000159 (benchmark.py:82)
[20117 ms][__main__] - CRITICAL : i 1901, Current time: 3.569683313369751, FPS: 136330.1369373088, Real Time Speedup: 1363.300458827981 (benchmark.py:82)
[20214 ms][__main__] - CRITICAL : i 1951, Current time: 3.6669156551361084, FPS: 136205.80212980547, Real Time Speedup: 1362.0566929101503 (benchmark.py:82)
[20331 ms][__main__] - CRITICAL : i 2001, Current time: 3.7841453552246094, FPS: 135368.828136962, Real Time Speedup: 1353.6875137737256 (benchmark.py:82)
[20429 ms][__main__] - CRITICAL : i 2051, Current time: 3.881354331970215, FPS: 135276.32626256172, Real Time Speedup: 1352.7624316700462 (benchmark.py:82)
[20537 ms][__main__] - CRITICAL : i 2101, Current time: 3.9898226261138916, FPS: 134806.82613061002, Real Time Speedup: 1348.0673751907805 (benchmark.py:82)
[20647 ms][__main__] - CRITICAL : i 2151, Current time: 4.099644184112549, FPS: 134317.84974911273, Real Time Speedup: 1343.1777163546326 (benchmark.py:82)
[20749 ms][__main__] - CRITICAL : i 2201, Current time: 4.201594114303589, FPS: 134105.12326082398, Real Time Speedup: 1341.0503194379417 (benchmark.py:82)
[20861 ms][__main__] - CRITICAL : i 2251, Current time: 4.314228057861328, FPS: 133570.9310896298, Real Time Speedup: 1335.7085727400472 (benchmark.py:82)
[20969 ms][__main__] - CRITICAL : i 2301, Current time: 4.421846866607666, FPS: 133214.80318853274, Real Time Speedup: 1332.14731361446 (benchmark.py:82)
[21075 ms][__main__] - CRITICAL : i 2351, Current time: 4.527986764907837, FPS: 132918.99045244386, Real Time Speedup: 1329.189064672864 (benchmark.py:82)
[21184 ms][__main__] - CRITICAL : i 2401, Current time: 4.637262582778931, FPS: 132547.02682766455, Real Time Speedup: 1325.469654952222 (benchmark.py:82)
[21289 ms][__main__] - CRITICAL : i 2451, Current time: 4.74210524559021, FPS: 132315.82300535776, Real Time Speedup: 1323.1576978593055 (benchmark.py:82)
[21391 ms][__main__] - CRITICAL : i 2501, Current time: 4.843557834625244, FPS: 132187.01749645625, Real Time Speedup: 1321.869589357423 (benchmark.py:82)
[21502 ms][__main__] - CRITICAL : i 2551, Current time: 4.954432487487793, FPS: 131812.35912932246, Real Time Speedup: 1318.122956982976 (benchmark.py:82)
[21604 ms][__main__] - CRITICAL : i 2601, Current time: 5.0566136837005615, FPS: 131680.11399240472, Real Time Speedup: 1316.8005190550693 (benchmark.py:82)
[21710 ms][__main__] - CRITICAL : i 2651, Current time: 5.1627278327941895, FPS: 131452.87268753047, Real Time Speedup: 1314.5281805228558 (benchmark.py:82)
[21818 ms][__main__] - CRITICAL : i 2701, Current time: 5.271094083786011, FPS: 131178.75552071742, Real Time Speedup: 1311.7870212027053 (benchmark.py:82)
[21922 ms][__main__] - CRITICAL : i 2751, Current time: 5.374696969985962, FPS: 131031.64892189781, Real Time Speedup: 1310.3159079705927 (benchmark.py:82)
[22021 ms][__main__] - CRITICAL : i 2801, Current time: 5.473724842071533, FPS: 130999.53455221458, Real Time Speedup: 1309.9946608107666 (benchmark.py:82)
[22122 ms][__main__] - CRITICAL : i 2851, Current time: 5.574838876724243, FPS: 130919.5573707905, Real Time Speedup: 1309.1949578160647 (benchmark.py:82)
[22225 ms][__main__] - CRITICAL : i 2901, Current time: 5.677508592605591, FPS: 130806.56139084442, Real Time Speedup: 1308.0650096761347 (benchmark.py:82)
[22325 ms][__main__] - CRITICAL : i 2951, Current time: 5.777953386306763, FPS: 130747.92356553332, Real Time Speedup: 1307.4786961442855 (benchmark.py:82)
[22429 ms][__main__] - CRITICAL : i 3001, Current time: 5.8815758228302, FPS: 130620.67572305734, Real Time Speedup: 1306.2061747915322 (benchmark.py:82)
[22534 ms][__main__] - CRITICAL : i 3051, Current time: 5.986496686935425, FPS: 130469.52998131442, Real Time Speedup: 1304.694728244214 (benchmark.py:82)
[22643 ms][__main__] - CRITICAL : i 3101, Current time: 6.095512866973877, FPS: 130236.01182690273, Real Time Speedup: 1302.359456046924 (benchmark.py:82)
[22754 ms][__main__] - CRITICAL : i 3151, Current time: 6.207106590270996, FPS: 129956.74692976769, Real Time Speedup: 1299.5669701268187 (benchmark.py:82)
[22855 ms][__main__] - CRITICAL : i 3201, Current time: 6.3081724643707275, FPS: 129903.75457941246, Real Time Speedup: 1299.037005723734 (benchmark.py:82)
[22970 ms][__main__] - CRITICAL : i 3251, Current time: 6.423028945922852, FPS: 129573.65245682672, Real Time Speedup: 1295.7359955032205 (benchmark.py:82)
[23075 ms][__main__] - CRITICAL : i 3301, Current time: 6.528048276901245, FPS: 129449.95687367783, Real Time Speedup: 1294.4991905134725 (benchmark.py:82)
[23183 ms][__main__] - CRITICAL : i 3351, Current time: 6.63611912727356, FPS: 129270.65647832978, Real Time Speedup: 1292.7060539042116 (benchmark.py:82)
[23293 ms][__main__] - CRITICAL : i 3401, Current time: 6.746131181716919, FPS: 129059.9866106565, Real Time Speedup: 1290.599455600951 (benchmark.py:82)
[23391 ms][__main__] - CRITICAL : i 3451, Current time: 6.843794345855713, FPS: 129088.54377645795, Real Time Speedup: 1290.884943086014 (benchmark.py:82)
[23502 ms][__main__] - CRITICAL : i 3501, Current time: 6.955066680908203, FPS: 128863.6344675391, Real Time Speedup: 1288.6357704109435 (benchmark.py:82)
[23612 ms][__main__] - CRITICAL : i 3551, Current time: 7.064771890640259, FPS: 128674.37973619516, Real Time Speedup: 1286.743145996584 (benchmark.py:82)
[23717 ms][__main__] - CRITICAL : i 3601, Current time: 7.169842481613159, FPS: 128574.02115020905, Real Time Speedup: 1285.7398694648975 (benchmark.py:82)
[23824 ms][__main__] - CRITICAL : i 3651, Current time: 7.2772510051727295, FPS: 128435.20634401016, Real Time Speedup: 1284.3515585026917 (benchmark.py:82)
[23932 ms][__main__] - CRITICAL : i 3701, Current time: 7.385110139846802, FPS: 128292.6675531762, Real Time Speedup: 1282.9263027736538 (benchmark.py:82)
[24045 ms][__main__] - CRITICAL : i 3751, Current time: 7.4975950717926025, FPS: 128075.09973772561, Real Time Speedup: 1280.7505086533854 (benchmark.py:82)
[24154 ms][__main__] - CRITICAL : i 3801, Current time: 7.6067821979522705, FPS: 127919.44132405588, Real Time Speedup: 1279.1939321171906 (benchmark.py:82)
[24265 ms][__main__] - CRITICAL : i 3851, Current time: 7.7183003425598145, FPS: 127729.61655393816, Real Time Speedup: 1277.2958498938046 (benchmark.py:82)
[24364 ms][__main__] - CRITICAL : i 3901, Current time: 7.8164567947387695, FPS: 127763.19352691459, Real Time Speedup: 1277.6316235054235 (benchmark.py:82)
[24485 ms][__main__] - CRITICAL : i 3951, Current time: 7.938199043273926, FPS: 127416.2220122623, Real Time Speedup: 1274.1617991681876 (benchmark.py:82)
[24601 ms][__main__] - CRITICAL : i 4001, Current time: 8.053448677062988, FPS: 127182.19779526319, Real Time Speedup: 1271.821601436055 (benchmark.py:82)
[24707 ms][__main__] - CRITICAL : i 4051, Current time: 8.160232782363892, FPS: 127086.49638594543, Real Time Speedup: 1270.8646296804998 (benchmark.py:82)
[24815 ms][__main__] - CRITICAL : i 4101, Current time: 8.267687797546387, FPS: 126982.94367858459, Real Time Speedup: 1269.829033981925 (benchmark.py:82)
[24927 ms][__main__] - CRITICAL : i 4151, Current time: 8.379844903945923, FPS: 126810.86672836641, Real Time Speedup: 1268.1083064889094 (benchmark.py:82)
[25033 ms][__main__] - CRITICAL : i 4201, Current time: 8.486179113388062, FPS: 126730.24009896602, Real Time Speedup: 1267.3020805469741 (benchmark.py:82)
[25145 ms][__main__] - CRITICAL : i 4251, Current time: 8.598006010055542, FPS: 126570.67526680637, Real Time Speedup: 1265.7062964017402 (benchmark.py:82)
[25260 ms][__main__] - CRITICAL : i 4301, Current time: 8.71290397644043, FPS: 126370.65671732296, Real Time Speedup: 1263.7062213747706 (benchmark.py:82)
[25386 ms][__main__] - CRITICAL : i 4351, Current time: 8.838768005371094, FPS: 126019.30817299368, Real Time Speedup: 1260.192741803367 (benchmark.py:82)
[25485 ms][__main__] - CRITICAL : i 4401, Current time: 8.938035488128662, FPS: 126051.80642289038, Real Time Speedup: 1260.5177952384897 (benchmark.py:82)
[25587 ms][__main__] - CRITICAL : i 4451, Current time: 9.040106058120728, FPS: 126044.48574526091, Real Time Speedup: 1260.444591514772 (benchmark.py:82)
[25698 ms][__main__] - CRITICAL : i 4501, Current time: 9.151201963424683, FPS: 125913.01166833403, Real Time Speedup: 1259.129788639243 (benchmark.py:82)
[25806 ms][__main__] - CRITICAL : i 4551, Current time: 9.258817195892334, FPS: 125831.99401580762, Real Time Speedup: 1258.3196161354479 (benchmark.py:82)
[25907 ms][__main__] - CRITICAL : i 4601, Current time: 9.359965562820435, FPS: 125839.72152987072, Real Time Speedup: 1258.396862703863 (benchmark.py:82)
[26023 ms][__main__] - CRITICAL : i 4651, Current time: 9.476065158843994, FPS: 125648.68869577924, Real Time Speedup: 1256.486349532072 (benchmark.py:82)
[26135 ms][__main__] - CRITICAL : i 4701, Current time: 9.587773084640503, FPS: 125519.81056532651, Real Time Speedup: 1255.1977935241212 (benchmark.py:82)
[26242 ms][__main__] - CRITICAL : i 4751, Current time: 9.694624662399292, FPS: 125456.67462320802, Real Time Speedup: 1254.566345138365 (benchmark.py:82)
[26350 ms][__main__] - CRITICAL : i 4801, Current time: 9.803276062011719, FPS: 125371.90876770356, Real Time Speedup: 1253.718782769053 (benchmark.py:82)
[26463 ms][__main__] - CRITICAL : i 4851, Current time: 9.915738344192505, FPS: 125240.84937272116, Real Time Speedup: 1252.4082528194638 (benchmark.py:82)
[26571 ms][__main__] - CRITICAL : i 4901, Current time: 10.024031639099121, FPS: 125164.75486343379, Real Time Speedup: 1251.6472509339294 (benchmark.py:82)
[26678 ms][__main__] - CRITICAL : i 4951, Current time: 10.130318403244019, FPS: 125115.05201157421, Real Time Speedup: 1251.1501373178437 (benchmark.py:82)
[26787 ms][__main__] - CRITICAL : i 5001, Current time: 10.239558696746826, FPS: 125030.32426549747, Real Time Speedup: 1250.3029515337366 (benchmark.py:82)
[26903 ms][__main__] - CRITICAL : i 5051, Current time: 10.355816841125488, FPS: 124862.70903652789, Real Time Speedup: 1248.6267741514614 (benchmark.py:82)
[27016 ms][__main__] - CRITICAL : i 5101, Current time: 10.469094276428223, FPS: 124734.32190038006, Real Time Speedup: 1247.3428497202428 (benchmark.py:82)
[27127 ms][__main__] - CRITICAL : i 5151, Current time: 10.57985782623291, FPS: 124638.3029149013, Real Time Speedup: 1246.3828044498043 (benchmark.py:82)
[27243 ms][__main__] - CRITICAL : i 5201, Current time: 10.695905208587646, FPS: 124482.71200414107, Real Time Speedup: 1244.8267870657503 (benchmark.py:82)
[27362 ms][__main__] - CRITICAL : i 5251, Current time: 10.814645051956177, FPS: 124299.52374984664, Real Time Speedup: 1242.9949086633633 (benchmark.py:82)
[27480 ms][__main__] - CRITICAL : i 5301, Current time: 10.932788133621216, FPS: 124127.07869205307, Real Time Speedup: 1241.2703538135077 (benchmark.py:82)
[27595 ms][__main__] - CRITICAL : i 5351, Current time: 11.04813528060913, FPS: 123989.73354826987, Real Time Speedup: 1239.8970946701813 (benchmark.py:82)
[27700 ms][__main__] - CRITICAL : i 5401, Current time: 11.153070449829102, FPS: 123970.83338060843, Real Time Speedup: 1239.7080952956296 (benchmark.py:82)
[27813 ms][__main__] - CRITICAL : i 5451, Current time: 11.266189336776733, FPS: 123862.21826632116, Real Time Speedup: 1238.6218419061315 (benchmark.py:82)
[27934 ms][__main__] - CRITICAL : i 5501, Current time: 11.38637661933899, FPS: 123678.95871715772, Real Time Speedup: 1236.7892505101504 (benchmark.py:82)
[28051 ms][__main__] - CRITICAL : i 5551, Current time: 11.503783941268921, FPS: 123529.3949472942, Real Time Speedup: 1235.293744659033 (benchmark.py:82)
[28171 ms][__main__] - CRITICAL : i 5601, Current time: 11.623569965362549, FPS: 123357.57568957159, Real Time Speedup: 1233.5755291718026 (benchmark.py:82)
[28288 ms][__main__] - CRITICAL : i 5651, Current time: 11.740862846374512, FPS: 123215.39807141789, Real Time Speedup: 1232.1536053991817 (benchmark.py:82)
[28417 ms][__main__] - CRITICAL : i 5701, Current time: 11.86977243423462, FPS: 122955.63309854253, Real Time Speedup: 1229.5560593173918 (benchmark.py:82)
[28538 ms][__main__] - CRITICAL : i 5751, Current time: 11.99120569229126, FPS: 122777.9179147135, Real Time Speedup: 1227.7788862072994 (benchmark.py:82)
[28653 ms][__main__] - CRITICAL : i 5801, Current time: 12.106031894683838, FPS: 122670.6861916413, Real Time Speedup: 1226.7065478495656 (benchmark.py:82)
[28760 ms][__main__] - CRITICAL : i 5851, Current time: 12.212596893310547, FPS: 122648.40039636602, Real Time Speedup: 1226.4837884691344 (benchmark.py:82)
[28866 ms][__main__] - CRITICAL : i 5901, Current time: 12.31847333908081, FPS: 122633.32599584373, Real Time Speedup: 1226.3329751369993 (benchmark.py:82)
[28990 ms][__main__] - CRITICAL : i 5951, Current time: 12.442890405654907, FPS: 122435.79530046518, Real Time Speedup: 1224.357601105575 (benchmark.py:82)
[29106 ms][__main__] - CRITICAL : i 6001, Current time: 12.558926820755005, FPS: 122323.77424822471, Real Time Speedup: 1223.2374870411074 (benchmark.py:82)
[29225 ms][__main__] - CRITICAL : i 6051, Current time: 12.677515029907227, FPS: 122189.18604388836, Real Time Speedup: 1221.89153872749 (benchmark.py:82)
[29347 ms][__main__] - CRITICAL : i 6101, Current time: 12.799669742584229, FPS: 122023.09154035735, Real Time Speedup: 1220.2305517373168 (benchmark.py:82)
[29473 ms][__main__] - CRITICAL : i 6151, Current time: 12.925378561019897, FPS: 121826.63831867445, Real Time Speedup: 1218.2661809400831 (benchmark.py:82)
[29583 ms][__main__] - CRITICAL : i 6201, Current time: 13.03618049621582, FPS: 121773.03579349464, Real Time Speedup: 1217.7300461403463 (benchmark.py:82)
[29702 ms][__main__] - CRITICAL : i 6251, Current time: 13.154721021652222, FPS: 121648.75608558222, Real Time Speedup: 1216.4873403774989 (benchmark.py:82)
[29822 ms][__main__] - CRITICAL : i 6301, Current time: 13.274776220321655, FPS: 121512.81014971873, Real Time Speedup: 1215.127795960964 (benchmark.py:82)
[29951 ms][__main__] - CRITICAL : i 6351, Current time: 13.40419316291809, FPS: 121294.51461599281, Real Time Speedup: 1212.9448009683445 (benchmark.py:82)
[30076 ms][__main__] - CRITICAL : i 6401, Current time: 13.529248714447021, FPS: 121119.45422126241, Real Time Speedup: 1211.1942433937386 (benchmark.py:82)
[30200 ms][__main__] - CRITICAL : i 6451, Current time: 13.653273344039917, FPS: 120956.73076090189, Real Time Speedup: 1209.567075268064 (benchmark.py:82)
[30323 ms][__main__] - CRITICAL : i 6501, Current time: 13.776039838790894, FPS: 120807.97118896637, Real Time Speedup: 1208.079523718276 (benchmark.py:82)
[30443 ms][__main__] - CRITICAL : i 6551, Current time: 13.895916938781738, FPS: 120686.89370023708, Real Time Speedup: 1206.8686264006221 (benchmark.py:82)
[30563 ms][__main__] - CRITICAL : i 6601, Current time: 14.015306234359741, FPS: 120572.1350688573, Real Time Speedup: 1205.7211660906787 (benchmark.py:82)
[30679 ms][__main__] - CRITICAL : i 6651, Current time: 14.13196587562561, FPS: 120482.5660247943, Real Time Speedup: 1204.8255179627893 (benchmark.py:82)
[30798 ms][__main__] - CRITICAL : i 6701, Current time: 14.250968217849731, FPS: 120374.58554410624, Real Time Speedup: 1203.7456137774404 (benchmark.py:82)
[30917 ms][__main__] - CRITICAL : i 6751, Current time: 14.369687795639038, FPS: 120270.91391489588, Real Time Speedup: 1202.7089795084244 (benchmark.py:82)
[31040 ms][__main__] - CRITICAL : i 6801, Current time: 14.493161916732788, FPS: 120129.43620576893, Real Time Speedup: 1201.2941051545022 (benchmark.py:82)
[31161 ms][__main__] - CRITICAL : i 6851, Current time: 14.61331844329834, FPS: 120017.58962758485, Real Time Speedup: 1200.1756417222603 (benchmark.py:82)
[31287 ms][__main__] - CRITICAL : i 6901, Current time: 14.739595413208008, FPS: 119857.78299289104, Real Time Speedup: 1198.577558504739 (benchmark.py:82)
[31412 ms][__main__] - CRITICAL : i 6951, Current time: 14.864301919937134, FPS: 119713.3438245173, Real Time Speedup: 1197.133169422495 (benchmark.py:82)
[31539 ms][__main__] - CRITICAL : i 7001, Current time: 14.991357564926147, FPS: 119552.56561292325, Real Time Speedup: 1195.5254469827994 (benchmark.py:82)
[31656 ms][__main__] - CRITICAL : i 7051, Current time: 15.109009027481079, FPS: 119468.82049062994, Real Time Speedup: 1194.687997533793 (benchmark.py:82)
[31771 ms][__main__] - CRITICAL : i 7101, Current time: 15.223893404006958, FPS: 119408.04103332397, Real Time Speedup: 1194.0801859301002 (benchmark.py:82)
[31876 ms][__main__] - CRITICAL : i 7151, Current time: 15.32851266860962, FPS: 119428.12427541464, Real Time Speedup: 1194.2810569965823 (benchmark.py:82)
[31985 ms][__main__] - CRITICAL : i 7201, Current time: 15.438118934631348, FPS: 119409.32396034746, Real Time Speedup: 1194.093018311857 (benchmark.py:82)
[32106 ms][__main__] - CRITICAL : i 7251, Current time: 15.558634757995605, FPS: 119307.07862882632, Real Time Speedup: 1193.0705486163217 (benchmark.py:82)
[32226 ms][__main__] - CRITICAL : i 7301, Current time: 15.67893934249878, FPS: 119208.02564011677, Real Time Speedup: 1192.080020748675 (benchmark.py:82)
[32344 ms][__main__] - CRITICAL : i 7351, Current time: 15.796459913253784, FPS: 119131.45731735993, Real Time Speedup: 1191.3143753859226 (benchmark.py:82)
[32463 ms][__main__] - CRITICAL : i 7401, Current time: 15.915514945983887, FPS: 119044.5520383328, Real Time Speedup: 1190.4452885520423 (benchmark.py:82)
[32586 ms][__main__] - CRITICAL : i 7451, Current time: 16.03922414779663, FPS: 118924.42201077362, Real Time Speedup: 1189.2440433299867 (benchmark.py:82)
[32704 ms][__main__] - CRITICAL : i 7501, Current time: 16.157124042510986, FPS: 118848.8284312004, Real Time Speedup: 1188.4880212476903 (benchmark.py:82)
[32828 ms][__main__] - CRITICAL : i 7551, Current time: 16.280396938323975, FPS: 118735.1468096971, Real Time Speedup: 1187.3512942151485 (benchmark.py:82)
[32937 ms][__main__] - CRITICAL : i 7601, Current time: 16.39005970954895, FPS: 118721.68405635146, Real Time Speedup: 1187.2166678646704 (benchmark.py:82)
[33050 ms][__main__] - CRITICAL : i 7651, Current time: 16.502952575683594, FPS: 118685.1570937255, Real Time Speedup: 1186.851399472571 (benchmark.py:82)
[33159 ms][__main__] - CRITICAL : i 7701, Current time: 16.6121187210083, FPS: 118675.73199638518, Real Time Speedup: 1186.7570304124577 (benchmark.py:82)
[33282 ms][__main__] - CRITICAL : i 7751, Current time: 16.734556674957275, FPS: 118572.33347402797, Real Time Speedup: 1185.7231827024784 (benchmark.py:82)
[33391 ms][__main__] - CRITICAL : i 7801, Current time: 16.84348201751709, FPS: 118565.47599997696, Real Time Speedup: 1185.6545753881774 (benchmark.py:82)
[33506 ms][__main__] - CRITICAL : i 7851, Current time: 16.95846152305603, FPS: 118516.37125875875, Real Time Speedup: 1185.1635459658025 (benchmark.py:82)
[33628 ms][__main__] - CRITICAL : i 7901, Current time: 17.081257820129395, FPS: 118413.71582832542, Real Time Speedup: 1184.1369103622883 (benchmark.py:82)
[33756 ms][__main__] - CRITICAL : i 7951, Current time: 17.20872163772583, FPS: 118280.44415169547, Real Time Speedup: 1182.8042284836306 (benchmark.py:82)
[33913 ms][__main__] - CRITICAL : i 8001, Current time: 17.36547875404358, FPS: 117949.82363386784, Real Time Speedup: 1179.4980258185017 (benchmark.py:82)
[34033 ms][__main__] - CRITICAL : i 8051, Current time: 17.485784769058228, FPS: 117870.33276211363, Real Time Speedup: 1178.7031026186717 (benchmark.py:82)
[34154 ms][__main__] - CRITICAL : i 8101, Current time: 17.606693983078003, FPS: 117787.89591826557, Real Time Speedup: 1177.8787518316713 (benchmark.py:82)
[34271 ms][__main__] - CRITICAL : i 8151, Current time: 17.72341799736023, FPS: 117734.3692345106, Real Time Speedup: 1177.3435022911406 (benchmark.py:82)
[34383 ms][__main__] - CRITICAL : i 8201, Current time: 17.835412979125977, FPS: 117712.74447606724, Real Time Speedup: 1177.1273031412736 (benchmark.py:82)
[34503 ms][__main__] - CRITICAL : i 8251, Current time: 17.955530881881714, FPS: 117638.14084195769, Real Time Speedup: 1176.381220975801 (benchmark.py:82)
[34624 ms][__main__] - CRITICAL : i 8301, Current time: 18.077091693878174, FPS: 117555.16182799246, Real Time Speedup: 1175.5514632366023 (benchmark.py:82)
[34736 ms][__main__] - CRITICAL : i 8351, Current time: 18.18875217437744, FPS: 117537.22405160072, Real Time Speedup: 1175.3721018548042 (benchmark.py:82)
[34843 ms][__main__] - CRITICAL : i 8401, Current time: 18.29590940475464, FPS: 117548.43227154513, Real Time Speedup: 1175.4841848532253 (benchmark.py:82)
[34953 ms][__main__] - CRITICAL : i 8451, Current time: 18.405831813812256, FPS: 117541.84550862222, Real Time Speedup: 1175.4182723779472 (benchmark.py:82)
[35066 ms][__main__] - CRITICAL : i 8501, Current time: 18.519179105758667, FPS: 117513.60334282732, Real Time Speedup: 1175.1358518818993 (benchmark.py:82)
[35187 ms][__main__] - CRITICAL : i 8551, Current time: 18.63939619064331, FPS: 117442.39038637851, Real Time Speedup: 1174.4237085754658 (benchmark.py:82)
[35300 ms][__main__] - CRITICAL : i 8601, Current time: 18.752327919006348, FPS: 117417.70907749135, Real Time Speedup: 1174.176926560599 (benchmark.py:82)
[35410 ms][__main__] - CRITICAL : i 8651, Current time: 18.863051891326904, FPS: 117406.90434148967, Real Time Speedup: 1174.0688653401758 (benchmark.py:82)
[35526 ms][__main__] - CRITICAL : i 8701, Current time: 18.979052543640137, FPS: 117363.88438341118, Real Time Speedup: 1173.6386669124167 (benchmark.py:82)
[35640 ms][__main__] - CRITICAL : i 8751, Current time: 19.093098878860474, FPS: 117333.25651847075, Real Time Speedup: 1173.332418668847 (benchmark.py:82)
[35767 ms][__main__] - CRITICAL : i 8801, Current time: 19.219846963882446, FPS: 117225.45624137673, Real Time Speedup: 1172.2543879146824 (benchmark.py:82)
[35885 ms][__main__] - CRITICAL : i 8851, Current time: 19.337337970733643, FPS: 117175.14813541778, Real Time Speedup: 1171.7513224367826 (benchmark.py:82)
[36000 ms][__main__] - CRITICAL : i 8901, Current time: 19.45306658744812, FPS: 117136.04739380463, Real Time Speedup: 1171.3602585936185 (benchmark.py:82)
[36131 ms][__main__] - CRITICAL : i 8951, Current time: 19.583836555480957, FPS: 117007.48113270095, Real Time Speedup: 1170.0746261448617 (benchmark.py:82)
[36251 ms][__main__] - CRITICAL : i 9001, Current time: 19.703779458999634, FPS: 116944.84680160583, Real Time Speedup: 1169.4482982101763 (benchmark.py:82)
[36365 ms][__main__] - CRITICAL : i 9051, Current time: 19.817887544631958, FPS: 116917.3790673459, Real Time Speedup: 1169.1736359506522 (benchmark.py:82)
[36477 ms][__main__] - CRITICAL : i 9101, Current time: 19.93029236793518, FPS: 116900.21957778384, Real Time Speedup: 1169.0020279659004 (benchmark.py:82)
[36592 ms][__main__] - CRITICAL : i 9151, Current time: 20.045067310333252, FPS: 116869.42121923884, Real Time Speedup: 1168.6940453852833 (benchmark.py:82)
[36707 ms][__main__] - CRITICAL : i 9201, Current time: 20.160168647766113, FPS: 116837.09158824259, Real Time Speedup: 1168.3707638909602 (benchmark.py:82)
[36825 ms][__main__] - CRITICAL : i 9251, Current time: 20.277817010879517, FPS: 116790.4534215442, Real Time Speedup: 1167.9043831661247 (benchmark.py:82)
[36941 ms][__main__] - CRITICAL : i 9301, Current time: 20.393884420394897, FPS: 116753.40480478593, Real Time Speedup: 1167.533911555122 (benchmark.py:82)
[37057 ms][__main__] - CRITICAL : i 9351, Current time: 20.510223865509033, FPS: 116715.22202035852, Real Time Speedup: 1167.1520573946411 (benchmark.py:82)
[37173 ms][__main__] - CRITICAL : i 9401, Current time: 20.62617039680481, FPS: 116679.70067596398, Real Time Speedup: 1166.7968449151656 (benchmark.py:82)
[37294 ms][__main__] - CRITICAL : i 9451, Current time: 20.746344089508057, FPS: 116620.81093780693, Real Time Speedup: 1166.207948552289 (benchmark.py:82)
[37423 ms][__main__] - CRITICAL : i 9501, Current time: 20.875402688980103, FPS: 116512.97928591359, Real Time Speedup: 1165.1296331754263 (benchmark.py:82)
[37545 ms][__main__] - CRITICAL : i 9551, Current time: 20.997796297073364, FPS: 116443.43102305646, Real Time Speedup: 1164.4341647939011 (benchmark.py:82)
[37665 ms][__main__] - CRITICAL : i 9601, Current time: 21.117368936538696, FPS: 116390.22820007708, Real Time Speedup: 1163.902124313046 (benchmark.py:82)
[37780 ms][__main__] - CRITICAL : i 9651, Current time: 21.23284649848938, FPS: 116360.06588119699, Real Time Speedup: 1163.6005150882786 (benchmark.py:82)
[37894 ms][__main__] - CRITICAL : i 9701, Current time: 21.346339225769043, FPS: 116341.04263793779, Real Time Speedup: 1163.4102574548097 (benchmark.py:82)
[38009 ms][__main__] - CRITICAL : i 9751, Current time: 21.461926221847534, FPS: 116310.87756587687, Real Time Speedup: 1163.1086206083917 (benchmark.py:82)
[38129 ms][__main__] - CRITICAL : i 9801, Current time: 21.581456422805786, FPS: 116259.7870889853, Real Time Speedup: 1162.597742453271 (benchmark.py:82)
[38250 ms][__main__] - CRITICAL : i 9851, Current time: 21.702834844589233, FPS: 116199.3579032878, Real Time Speedup: 1161.9934513809967 (benchmark.py:82)


## Depth Only (new code)

This is rendering the depth at each step.

docker compose -f docker/docker-compose.yaml run --rm \
  -e AERIAL_GYM_BENCH_ENVS=1,2,4,8,16,32 \
  -e AERIAL_GYM_BENCH_WARMUP_STEPS=100 \
  -e AERIAL_GYM_BENCH_STEPS=300 \
  -e AERIAL_GYM_BENCH_HEADLESS=1 \
  aerialgym python aerial_gym/examples/benchmark_matterport_depth_only.py

[13068 ms][__main__] - WARNING : Running depth-only benchmark: env=env_with_obstacles, robot=base_quadrotor_with_camera, env_counts=[1, 2, 4, 8, 16, 32], warmup=100, bench=300, res=240x135, max_range=80.0 (benchmark_matterport_depth_only.py:384)
[13068 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_only.py:398)
[32057 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_only.py:398)
[50829 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_only.py:398)
[71396 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_only.py:398)
[94517 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_only.py:398)
[124797 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_only.py:398)
[174294 ms][__main__] - WARNING : 
=== Vanilla Parallel Benchmark: Depth-only === (benchmark_matterport_depth_only.py:309)
  envs |    depth FPS |    FPS/env |  depth RTF |    RTF/env |   elapsed(s)
----------------------------------------------------------------------------
     1 |        38.07 |      38.07 |       0.38 |       0.38 |        7.881
     2 |        68.04 |      34.02 |       0.68 |       0.34 |        8.818
     4 |       121.83 |      30.46 |       1.22 |       0.30 |        9.850
     8 |       213.63 |      26.70 |       2.14 |       0.27 |       11.234
    16 |       284.56 |      17.78 |       2.85 |       0.18 |       16.868
    32 |       301.44 |       9.42 |       3.01 |       0.09 |       31.847


## Depth Only (old code)

This is rendering the depth at each step.

docker compose -f docker/docker-compose.yaml run --rm \
  -e AERIAL_GYM_BENCH_ENVS=1,2,4,8,16,32 \
  -e AERIAL_GYM_BENCH_WARMUP_STEPS=100 \
  -e AERIAL_GYM_BENCH_STEPS=300 \
  -e AERIAL_GYM_BENCH_HEADLESS=1 \
  aerialgym python aerial_gym/examples/benchmark_matterport_depth_only.py

[12455 ms][__main__] - WARNING : Running depth-only benchmark: env=env_with_obstacles, robot=base_quadrotor_with_camera, env_counts=[1, 2, 4, 8, 16, 32], warmup=100, bench=300, res=240x135, max_range=80.0 (benchmark_matterport_depth_only.py:384)
[12455 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_only.py:398)
[32979 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_only.py:398)
[54344 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_only.py:398)
[77624 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_only.py:398)
[105253 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_only.py:398)
[143689 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_only.py:398)
[210144 ms][__main__] - WARNING : 
=== Vanilla Parallel Benchmark: Depth-only === (benchmark_matterport_depth_only.py:309)
  envs |    depth FPS |    FPS/env |  depth RTF |    RTF/env |   elapsed(s)
----------------------------------------------------------------------------
     1 |        33.99 |      33.99 |       0.34 |       0.34 |        8.827
     2 |        58.66 |      29.33 |       0.59 |       0.29 |       10.229
     4 |       106.16 |      26.54 |       1.06 |       0.27 |       11.304
     8 |       162.38 |      20.30 |       1.62 |       0.20 |       14.780
    16 |       215.17 |      13.45 |       2.15 |       0.13 |       22.308
    32 |       212.43 |       6.64 |       2.12 |       0.07 |       45.192

## Depth vs RGB (25 FPS sim time rendering - gif off for maximum speed - almost empty scene)

docker compose -f docker/docker-compose.yaml run --rm \
  -e AERIAL_GYM_BENCH_ENVS=1,2,4,8,16,32 \
  -e AERIAL_GYM_BENCH_WARMUP_STEPS=100 \
  -e AERIAL_GYM_BENCH_STEPS=300 \
  -e AERIAL_GYM_BENCH_HEADLESS=1 \
  -e AERIAL_GYM_BENCH_RENDER_EVERY=4 \
  aerialgym python aerial_gym/examples/benchmark_matterport_depth_vs_rgbd.py

[14186 ms][__main__] - WARNING : Running benchmark on env counts [1, 2, 4, 8, 16, 32], warmup=100, bench=300, resolution=240x135, max_range=80.0 (benchmark_matterport_depth_vs_rgbd.py:457)
[14186 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:471)
[33140 ms][__main__] - WARNING : Shaded RGBD case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:493)
[51362 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:471)
[69570 ms][__main__] - WARNING : Shaded RGBD case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:493)
[88388 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:471)
[108659 ms][__main__] - WARNING : Shaded RGBD case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:493)
[130011 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:471)
[153857 ms][__main__] - WARNING : Shaded RGBD case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:493)
[177845 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:471)
[210693 ms][__main__] - WARNING : Shaded RGBD case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:493)
[241292 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:471)
[283480 ms][__main__] - WARNING : Shaded RGBD case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:493)
[325635 ms][__main__] - WARNING : 
=== Matterport Parallel Benchmark: Depth-only vs RGBD === (benchmark_matterport_depth_vs_rgbd.py:372)
  envs | depth stepFPS |  rgbd stepFPS | depth renderFPS |  rgbd renderFPS |  rgbd/depth
------------------------------------------------------------------------------------------------
     1 |        331.35 |        306.37 |           82.84 |           76.59 |       0.925
     2 |        591.70 |        540.05 |          147.92 |          135.01 |       0.913
     4 |       1063.11 |        987.00 |          265.78 |          246.75 |       0.928
     8 |       2024.33 |       1986.73 |          506.08 |          496.68 |       0.981
    16 |       3752.42 |       3009.67 |          938.10 |          752.42 |       0.802
    32 |       7878.79 |       7491.88 |         1969.70 |         1872.97 |       0.951

## Depth vs RGB (25 FPS sim time rendering - gif off for maximum speed - full scene)

docker compose -f docker/docker-compose.yaml run --rm  \
  -e AERIAL_GYM_BENCH_ENVS=1,2,4,8,16,32 \
  -e AERIAL_GYM_BENCH_WARMUP_STEPS=100 \
  -e AERIAL_GYM_BENCH_STEPS=300 \
  -e AERIAL_GYM_BENCH_HEADLESS=1 \
  -e AERIAL_GYM_BENCH_SPAWN_CENTER=-9.0,1.6,1.6 \
  -e AERIAL_GYM_BENCH_SPAWN_BOUNDS=0.3,0.3,0.3 \
  -e AERIAL_GYM_BENCH_RENDER_EVERY=4 \
  -e AERIAL_GYM_BENCH_SAVE_GIFS=0 \
  aerialgym python aerial_gym/examples/benchmark_matterport_depth_vs_rgbd.py

[13325 ms][__main__] - WARNING : Configured benchmark spawn region from AERIAL_GYM_BENCH: lower=[-9.3  1.3  1.3] upper=[-8.7  1.9  1.9] (benchmark_matterport_depth_vs_rgbd.py:116)
[13325 ms][__main__] - WARNING : Running benchmark on env counts [1, 2, 4, 8, 16, 32], warmup=100, bench=300, resolution=240x135, max_range=80.0 (benchmark_matterport_depth_vs_rgbd.py:459)
[13325 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:473)
[33222 ms][__main__] - WARNING : Shaded RGBD case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:495)
[52740 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:473)
[72546 ms][__main__] - WARNING : Shaded RGBD case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:495)
[92947 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:473)
[114754 ms][__main__] - WARNING : Shaded RGBD case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:495)
[137927 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:473)
[164809 ms][__main__] - WARNING : Shaded RGBD case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:495)
[190933 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:473)
[227150 ms][__main__] - WARNING : Shaded RGBD case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:495)
[261427 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:473)
[310806 ms][__main__] - WARNING : Shaded RGBD case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:495)
[357619 ms][__main__] - WARNING : 
=== Matterport Parallel Benchmark: Depth-only vs RGBD === (benchmark_matterport_depth_vs_rgbd.py:372)
  envs |    depth FPS |     rgbd FPS |  d FPS/env |  r FPS/env |  rgbd/depth |  depth RTF |   rgbd RTF
--------------------------------------------------------------------------------------------------------------
     1 |       285.81 |       229.28 |     285.81 |     229.28 |       0.802 |       2.86 |       2.29
     2 |       523.95 |       475.99 |     261.98 |     238.00 |       0.908 |       5.24 |       4.76
     4 |      1005.73 |       882.16 |     251.43 |     220.54 |       0.877 |      10.06 |       8.82
     8 |      1233.55 |      1281.61 |     154.19 |     160.20 |       1.039 |      12.34 |      12.82
    16 |       798.08 |      1203.84 |      49.88 |      75.24 |       1.508 |       7.98 |      12.04
    32 |      1492.04 |      2129.26 |      46.63 |      66.54 |       1.427 |      14.92 |      21.29


[13729 ms][__main__] - WARNING : Configured benchmark spawn region from AERIAL_GYM_BENCH: lower=[-9.3  1.3  1.3] upper=[-8.7  1.9  1.9] (benchmark_matterport_depth_vs_rgbd.py:116)
[13729 ms][__main__] - WARNING : Running benchmark on env counts [1, 2, 4, 8, 16, 32], warmup=100, bench=300, resolution=240x135, max_range=80.0 (benchmark_matterport_depth_vs_rgbd.py:459)
[13729 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:473)
[34658 ms][__main__] - WARNING : Shaded RGBD case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:495)
[54546 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:473)
[75132 ms][__main__] - WARNING : Shaded RGBD case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:495)
[95621 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:473)
[117278 ms][__main__] - WARNING : Shaded RGBD case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:495)
[140293 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:473)
[165768 ms][__main__] - WARNING : Shaded RGBD case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:495)
[190812 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:473)
[224790 ms][__main__] - WARNING : Shaded RGBD case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:495)
[258729 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:473)
[308821 ms][__main__] - WARNING : Shaded RGBD case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:495)
[357495 ms][__main__] - WARNING : 
=== Matterport Parallel Benchmark: Depth-only vs RGBD === (benchmark_matterport_depth_vs_rgbd.py:372)
  envs |    depth FPS |     rgbd FPS |  d FPS/env |  r FPS/env |  rgbd/depth |  depth RTF |   rgbd RTF
--------------------------------------------------------------------------------------------------------------
     1 |       248.79 |       276.13 |     248.79 |     276.13 |       1.110 |       2.49 |       2.76
     2 |       457.51 |       517.45 |     228.75 |     258.72 |       1.131 |       4.58 |       5.17
     4 |       888.84 |       880.52 |     222.21 |     220.13 |       0.991 |       8.89 |       8.81
     8 |      1219.28 |      1187.81 |     152.41 |     148.48 |       0.974 |      12.19 |      11.88
    16 |      1095.61 |       988.93 |      68.48 |      61.81 |       0.903 |      10.96 |       9.89
    32 |      1339.78 |      1718.79 |      41.87 |      53.71 |       1.283 |      13.40 |      17.19


## Depth vs RGB (25 FPS sim time rendering - gif off for maximum speed - full scene - higher res)

docker compose -f docker/docker-compose.yaml run --rm  \
  -e AERIAL_GYM_BENCH_ENVS=1,2,4,8,16,32 \
  -e AERIAL_GYM_BENCH_WARMUP_STEPS=100 \
  -e AERIAL_GYM_BENCH_STEPS=300 \
  -e AERIAL_GYM_BENCH_HEADLESS=1 \
  -e AERIAL_GYM_BENCH_SPAWN_CENTER=-9.0,1.6,1.6 \
  -e AERIAL_GYM_BENCH_SPAWN_BOUNDS=0.3,0.3,0.3 \
  -e AERIAL_GYM_BENCH_RENDER_EVERY=4 \
  -e AERIAL_GYM_BENCH_SAVE_GIFS=0 \
  -e AERIAL_GYM_BENCH_CAM_WIDTH=480 \
  -e AERIAL_GYM_BENCH_CAM_HEIGHT=270 \
  aerialgym python aerial_gym/examples/benchmark_matterport_depth_vs_rgbd.py

  [14298 ms][__main__] - WARNING : Configured benchmark spawn region from AERIAL_GYM_BENCH: lower=[-9.3  1.3  1.3] upper=[-8.7  1.9  1.9] (benchmark_matterport_depth_vs_rgbd.py:116)
[14298 ms][__main__] - WARNING : Running benchmark on env counts [1, 2, 4, 8, 16, 32], warmup=100, bench=300, resolution=480x270, max_range=80.0 (benchmark_matterport_depth_vs_rgbd.py:459)
[14298 ms][__main__] - WARNING : Depth-only case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:473)
[33272 ms][__main__] - WARNING : Shaded RGBD case: num_envs=1 (benchmark_matterport_depth_vs_rgbd.py:495)
[52070 ms][__main__] - WARNING : Depth-only case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:473)
[71491 ms][__main__] - WARNING : Shaded RGBD case: num_envs=2 (benchmark_matterport_depth_vs_rgbd.py:495)
[91196 ms][__main__] - WARNING : Depth-only case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:473)
[114563 ms][__main__] - WARNING : Shaded RGBD case: num_envs=4 (benchmark_matterport_depth_vs_rgbd.py:495)
[137351 ms][__main__] - WARNING : Depth-only case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:473)
[165542 ms][__main__] - WARNING : Shaded RGBD case: num_envs=8 (benchmark_matterport_depth_vs_rgbd.py:495)
[195127 ms][__main__] - WARNING : Depth-only case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:473)
[236425 ms][__main__] - WARNING : Shaded RGBD case: num_envs=16 (benchmark_matterport_depth_vs_rgbd.py:495)
[279365 ms][__main__] - WARNING : Depth-only case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:473)
[341509 ms][__main__] - WARNING : Shaded RGBD case: num_envs=32 (benchmark_matterport_depth_vs_rgbd.py:495)
[400939 ms][__main__] - WARNING : 
=== Matterport Parallel Benchmark: Depth-only vs RGBD === (benchmark_matterport_depth_vs_rgbd.py:372)
  envs |    depth FPS |     rgbd FPS |  d FPS/env |  r FPS/env |  rgbd/depth |  depth RTF |   rgbd RTF
--------------------------------------------------------------------------------------------------------------
     1 |       263.09 |       265.56 |     263.09 |     265.56 |       1.009 |       2.63 |       2.66
     2 |       358.87 |       297.15 |     179.43 |     148.58 |       0.828 |       3.59 |       2.97
     4 |       316.38 |       470.48 |      79.10 |     117.62 |       1.487 |       3.16 |       4.70
     8 |       527.49 |       364.79 |      65.94 |      45.60 |       0.692 |       5.27 |       3.65
    16 |       437.22 |       416.69 |      27.33 |      26.04 |       0.953 |       4.37 |       4.17
    32 |       606.65 |       609.12 |      18.96 |      19.03 |       1.004 |       6.07 |       6.09