# Deep Reinforcement Learning hands on by Minecraft with ChainerRL and Azure Notebooks

## Minecraft

Do you know Minecraft? 

[![PR movie](https://img.youtube.com/vi/MmB9b5njVbA/0.jpg)](https://www.youtube.com/watch?v=MmB9b5njVbA)

[Official Wiki](https://minecraft.gamepedia.com/Minecraft_Wiki) says as belows:

> [![Minecraft](https://gamepedia.cursecdn.com/minecraft_gamepedia/4/4d/Mclogo.svg?version=efb6f83c8e7d7bd4cdf0b73a2826c393)](https://minecraft.gamepedia.com/Minecraft) is a [sandbox](https://en.wikipedia.org/wiki/Open_world) construction game created by [Mojang AB](https://minecraft.gamepedia.com/Mojang_AB) founder [Markus "Notch" Persson](https://minecraft.gamepedia.com/Markus_Persson), inspired by *Infiniminer*, *Dwarf Fortress*, *Dungeon Keeper*, and Notch's past games *Legend of the Chambered* and *RubyDung*. Gameplay involves [players](https://minecraft.gamepedia.com/Player) interacting with the game world by placing and breaking various types of [blocks](https://minecraft.gamepedia.com/Block) in a [three-dimensional environment](https://minecraft.gamepedia.com/Overworld). In this environment, players can build creative structures, creations, and artwork on [multiplayer](https://minecraft.gamepedia.com/Multiplayer) servers and singleplayer worlds across multiple [game modes](https://minecraft.gamepedia.com/Gameplay).

This game has some characteristics related to today's reinforcement learning.

- can freely make buildings and other things
- can play various games
- can multiplay

These characteristics made Minecraft good material for studying reinforcement learning.

## Marlo Project

In this hands-on, you do deep reinforcement learning with Minecraft as a simulator environment. 
This time you use a Minecraft environment called [marLo](https://github.com/crowdAI/marLo) which is used in [MARLO](https://www.crowdai.org/challenges/marlo-2018), a contest utilizing Minecraft for a deep reinforcement learning. 
You can easily use a reinforcement learning framework such as ChainerRL because marLo is compatible with [OpenAI's Gym](https://github.com/openai/gym)(although it is not complete ... for example wrapper used for saving movies cannot be used).

MarLo has some environments as follows. For example, you can make an AI walking on a single road on lava with deep reinforcement learning. Now you learn with `MarLo-FindTheGoal-v0` environment, and then you try assignments.

| `MarLo-MazeRunner-v0`![Alt text](https://camo.qiitausercontent.com/19c8cc6ab8297d62c787c0b5ab41859b810685b6/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f753435664e517847353977666e52707a774a2f67697068792e676966) | `MarLo-CliffWalking-v0`![Alt text](https://camo.qiitausercontent.com/a1835ac77773d3bc0d668bc35747d133dfd0afbb/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f6566346c50474e71614c6c4b7234357257422f67697068792e676966) | `MarLo-CatchTheMob-v0`![Alt text](https://camo.qiitausercontent.com/54f211227c0e9b47979a9c78a8638fed2041eda7/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f39413167485a72576361533441597a6349552f67697068792e676966) | `MarLo-FindTheGoal-v0`![Alt text](https://camo.qiitausercontent.com/66ae6f6fc759a2ada4198cb45275597d5da96ad3/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f3167576b51624473484f666f346b5a585a762f67697068792e676966) |
| :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| `MarLo-Attic-v0`![Alt text](https://camo.qiitausercontent.com/96c3cf42cd5342972f1b99b8c3e085ec8f19836c/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f34374337415942334641366b67724d6951332f67697068792e676966) | `MarLo-DefaultFlatWorld-v0`![Alt text](https://camo.qiitausercontent.com/ce615eddc9c3c7af3dab8fae20762ca85646dc30/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f4c307339515875523676494a6836413064712f67697068792e676966) | `MarLo-DefaultWorld-v0`![Alt text](https://camo.qiitausercontent.com/8ac767a629ef5f6a1f78e80724819e123fd68e72/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f344e78376759694d394e44724d724d616f372f67697068792e676966) | `MarLo-Eating-v0`![Alt text](https://camo.qiitausercontent.com/6feb3c4dff76239afac2a1ebd112c9ea0dc9dfab/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f704f624e4d6a6a66634749357456686d58362f67697068792e676966) |
| `MarLo-Obstacles-v0`![Alt text](https://camo.qiitausercontent.com/5579a7467bf3424ed4f8c60f6e9cad20c769abaf/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f3573596d46466b713761454d4b54624b50342f67697068792e676966) | `MarLo-TrickyArena-v0`![Alt text](https://camo.qiitausercontent.com/9736b6d4e10198781bf4191dee9c629fcc54de94/68747470733a2f2f6d656469612e67697068792e636f6d2f6d656469612f316731627877326e44334739667a325756562f67697068792e676966) |                                                              |                                                              |



## Hands-on content

This hands-on is based on [marlo-handson](https://qiita.com/keisuke-umezawa/items/fcf5d00474e244217a5e).

## Requirements

As of December 14, 2018, the following is necessary.

Python 3.5+ environment with

- Chainer v5.0.0
- CuPy v5.0.0
- ChainerRL v0.4.0
- marlo v0.0.1.dev23

## Environment construction by Azure

\* It is necessary to sign up Azure pay-as-you-go subscription to use the GPU in the following procedure. If you use CPU only, Azure free trial version is fine.

### 1. Create a VM using Azure Cloud Shell

##### Access [Azure Portal](https://portal.azure.com/#home)

1. If you do not have a Microsoft account, please create one.
   If you have never used Azure before, sign up for a free account [here](https://azure.microsoft.com/ja-jp/free/).

   

2. Check the upper right, and make sure that it is the directory you will use this time.
   
   

   ![Alt text](https://camo.qiitausercontent.com/a9c16ac481f33b6159245081cb57e013383a5fd0/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f30626430653365312d666431662d626639332d613438642d3538373636656665663135322e706e67)

   \* If you have created an account personally, there is no problem with the default directory.

   

3. Launch Cloud Shell

   

   ![Alt text](https://camo.qiitausercontent.com/33ec4c11ab83a90f967592e3326dade188f7605a/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f63666633333834612d376564642d313234322d653431612d6130323532356630616632362e706e67)

   

4. Click Bash

   Please skip this step if you have used Cloud Shell.

   

   ![](https://camo.qiitausercontent.com/9c9dc27334fabb2887334a668efe8308bd730374/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f32373461336665322d323330632d376261342d373931312d3030383033353832636338392e706e67)

   

   If you have not signed up for Azure, the following screen will appear. Click "Create Azure Subscription" and register for a free trial.

   

   ![](https://camo.qiitausercontent.com/bcc34655247e2da884573a693e17b1a1fbecca4b/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f63366537306338312d633633632d653833612d653833662d6135663064636433396238382e706e67)

   

   In addition, when there is no storage, it is displayed as follows. If you have a free trial version, select it and click Create Storage.

   

   ![](https://camo.qiitausercontent.com/a16d5734f8b39cd396a8fbaba937884c96bb7ebf/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f31323966653730382d653439612d396535302d366264322d3665323232396236343037662e706e67)

   

5. The Cloudshell console opens. Click □ and keep the console screen larger.

   

   ![](https://camo.qiitausercontent.com/9d1f5c508ed13561e30c33454ea6cd03957d40c6/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f31643032303065322d356138652d363565612d643862662d3232313736316365306166632e706e67)

   

6. Create resource group

   \* You can skip this step if you have already created a resource group and use it.* 

   You can create resource groups and organize resources (VMs, networks, etc.) to be used this time.
   Create a resource group named "malmo" as follows: (If you like another name, replace "malmo" with the name in the following steps)

   ```bash
   az group create -g malmo -l eastus
   ```

   If successful, you will see something like this:

   ![](https://camo.qiitausercontent.com/92b22ef7f994d8fc1aaec76b505677c866c556f6/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f38373565313561382d353265622d323031392d313937342d3366636330303332363362642e706e67)

   

7. Generate password

   Enter the following to generate a password.

   ```bash
   echo user name: ${USER}
   VM_PW=`openssl rand -base64 12`
   echo VM password: ${VM_PW}
   ```

   Make a note of the username and password.

   

8. Create VM

   Please execute with Cloud Shell as follows to create with CPU instance.
   From this point on, when you launch an instance, there will be a cost. Stop when it's no longer needed.
   Generally, it is around $35 a month (reference: [price](https://azure.microsoft.com/pricing/details/virtual-machines/linux/))

   \* Storage for Cloud Shell, VM storage, etc. cost slightly even when not started.

   ```bash
   az vm create --location eastus --resource-group malmo --name ${USER}-vm \
    --admin-username ${USER} --admin-password ${VM_PW} \
    --authentication-type password \
    --public-ip-address-dns-name ${USER}  \
    --image microsoft-ads:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest \
    --size Standard_B2s
   ```

   \* **The following is unnecessary if you use CPU instance.**
   To create on a GPU instance please execute as follows.

   ```bash
   az vm create --location eastus --resource-group malmo --name ${USER}-vm \
    --admin-username ${USER} --admin-password ${VM_PW} \
    --authentication-type password \
    --public-ip-address-dns-name ${USER}  \
    --image microsoft-ads:linux-data-science-vm-ubuntu:linuxdsvmubuntu:latest \
    --size Standard_NC6
   ```

   Please keep in mind that it will cost $600 or more if you keep running Standard_NC6 of GPU instance for 1 month. (Reference : [price](https://azure.microsoft.com/pricing/details/virtual-machines/linux/))
   Also, it can not be used in the free trial version.

   If it succeeds, it will be displayed as follows. Make a note of the following `publicIpAddress`.

   ![](https://camo.qiitausercontent.com/2adfb1afdda121c19a19ba5077795af54e4261f4/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f66353138636364322d666130612d306133392d306233352d3536336630653866303335362e706e67)

   You can connect using SSH by the above IP address and the password you created initially.
   In this hands-on, we use Azure Notebooks to make later work easier. (see below)

   If you get an error here, there are the following possibilities.

   - VM name (`--name $ {USER} -VM)` and DNS name (`--public-ip-address-dns-name $ {USER}`) are already in use (user name in the same resource group Is duplicated)
   - - Please replace `--name $ {USER} -VM` with `--name $ {USER} -VM2` and `--public-ip-address-dns-name $ {USER}` with `--public-ip-address-dns-name $ {USER}-2`.
   - You are using a subscription for which Standard_NC6 can not be used (cannot be used with the free trial version)

9. Open port

   Three ports 8000 and 8001 6080 are opened to connect from Azure Notebooks and to connect to a web server called noVNC to display the screen.
   Execute the following command in Cloud Shell.

   ```bash
   az vm open-port --resource-group malmo --name ${USER}-VM --port 8000 --priority 1010
   az vm open-port --resource-group malmo --name ${USER}-VM --port 8001 --priority 1020
   az vm open-port --resource-group malmo --name ${USER}-VM --port 6080 --priority 1030
   ```

   

### 2. Connect Azure Notebooks

1. Clone Notebook

   Please move [here](https://notebooks.azure.com/ikeyasu/projects/marlo) and click Clone.

   

   ![](https://camo.qiitausercontent.com/56a29f47f4951d09deeafe39f9aa29b562f807b3/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f34663166643232632d666663392d376438312d626465322d6136633831373863623266662e706e67)

   

2. Direct Compute settings

   Configure Direct Compute to connect to the VM created earlier.

   Click ▽ on the top left and click "Direct Compute" to display the setting screen.

   

   ![](https://camo.qiitausercontent.com/16b4f964a153ba31b038640474bc5cff6b3240be/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f31393362393839332d646164312d303430312d303565332d3437643238653466633138642e706e67)

   Enter the IP you noted earlier, the username and password here. Any name is acceptable. Press Validate. (Port can not be changed.)

   ![](https://camo.qiitausercontent.com/df1068b9b9dcb7aef2aacc5e132c4704e872d69c/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f66306433643033622d623732312d333333622d313461632d3532323134396666363937302e706e67)

   

   Here, errors may occur as follows. In this case, press Validate several times.

   

   ![](https://camo.qiitausercontent.com/c829217e22bd075b89bb473bde83984cbc114810/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f61633664386632372d656263652d663763302d373366382d3663656364323132313630372e706e67)

   

   Press Run if you see something like this screen.

   ![](https://camo.qiitausercontent.com/90d33ab7ea91a2319eda01490e028beef8c4413c/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f35373036356261322d653862352d386462392d373031662d6166656232366533643334392e706e67)

3. Open notebook

   Please press marlo.ipynb.

   

   ![](https://camo.qiitausercontent.com/b927df1a6886745ba5b9a616efd43283baa9699b/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f35373432643363392d366634322d663864332d343733352d3965303133313535636331352e706e67)

   

   Then, you'll see the following screen.

   

   ![](https://camo.qiitausercontent.com/6d7d8335be72adbace1571f1490271b623a12e4f/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f32633939333662612d383464322d303438612d333263612d6136313864386234326639312e706e67)

   

### 3. Run Notebook

The screen opened above is a system called Jupyter Notebook.
Each square is called a "cell" and can be executed with SHIFT + ENTER.
Let's run the top cell.

The left `In []` part becomes `In [*]`. This means running.
It looks like `In [1]`, and when the output is displayed, execution is complete.

**When you're done, don't forget to stop or remove the VM.**



## VM stop / start / delete

Please stop VM when unnecessary. You can start later with the same VM.
If it does not stop, it will cost you.
There are two ways to stop and delete; 1:Cloud Shell 2:Azure Portal.

### 1. Use Cloud Shell

#### Stop VM

You can stop the VM as follows:

```bash
azure vm deallocate --resource-group malmo --name ${USER}-vm
```

#### Resume a stopped VM

You can start the VM as follows: 

```bash
azure vm start --resource-group malmo --name ${USER}-vm
```

**Please note that the IP address will change.** You can check the IP as follows.

```bash
az vm list-ip-addresses --output table --name ${USER}-vm
```

#### Delete VM

For deletion, it is more convenient to use Azure Portal. Please refer to the following.

### 2. Use Azure Portal

![](https://camo.qiitausercontent.com/f4fa548bf209f6b13fda6d699fd4c17483c07df3/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f33666335356632392d666232662d336264302d333737332d3064303561333631366332392e706e67)



#### Stop / start using Azure Portal

Click below to display the VM list.



![](https://camo.qiitausercontent.com/37857cff083104fc386b4fe86df4554caec33946/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f63363061393161362d383833392d303962662d653438632d3732333062326632346331622e706e67)



Click on your VM from the list. If the list is large, enter the name of your VM in "Filter by Name".



![](https://camo.qiitausercontent.com/b9b1ea0930c6e73e6856d407f34403b3591777c1/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f34616362306135652d636162312d316365632d383232352d3865336138666564333835632e706e67)



You can choose to stop or start.



![](https://camo.qiitausercontent.com/c1118d35000519436ab2f8fde4053bea338e09c7/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f36323337666530382d383366642d303731382d333935612d3434353731666163316463362e706e67)



#### Delete VM

When you create a VM, multiple resources such as disks are created. Click on "All Resources".



![](https://camo.qiitausercontent.com/8b76df373d4bfb38ad0b5cfcc9f4ef6e822f48c8/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f37373233646435362d356263652d356636352d666466612d3134646365633332646663652e706e67)



Click "Filter by Name" and enter your VM name to view resources related to your VM.



![](https://camo.qiitausercontent.com/4f9edb84fdfdc929c7a26100083b3b8d4f175ff9/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f36643862373665622d663338322d666565312d643366352d6331623731643834633237362e706e67)



Check the □ next to Name to select all and press Delete.



![](https://camo.qiitausercontent.com/1a2bc5e59189262ed71ea1b7d3f1062363bb393c/68747470733a2f2f71696974612d696d6167652d73746f72652e73332e616d617a6f6e6177732e636f6d2f302f33383535352f39663463313661302d343933652d613135652d653732632d3631313034323762623238622e706e67)

As confirmation is given, please enter yes and delete it.

**Be careful not to delete irrelevant resources! **

------

This tutorial is a English version of [original hands-on in japanese](https://qiita.com/ikeyasu/items/d6e587126a54ce559604) (Allowed by the author).
