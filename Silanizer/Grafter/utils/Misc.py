from logging import raiseExceptions
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import Silanizer
import Silanizer.Grafter.utils as Utils
import os

def silanize(path,grafDens,beads):

    surfDist=0.5
    surfGeometry="cylindrical"
    dep_res=['SIL']
    surf_res=['SI', 'LAY']
    bulk_res=['BULK']
    sil_norm="y"
    matrix_name=f"{path}/hole_matrix_4nm.gro"

    # for the "deposit" residue, which should be the molecules to attach, count the number of beads for each molecule
    nBeads, nSils = 0,0
    for dr in dep_res:
        nBeads+=len(beads[dr])

    # tests if there are molecules to attach to surface and run routines in case positive or just builds surface in case negative
    if nBeads > 0:

        # checks surface geometry
        if surfGeometry == "flat":
            # creates the fcc structure with given parameters if no surface file is given
            if not matrix_name: 
                nx,ny,nz = 50, 10, 9
                a = 0.47*2**(1/6) #lattice space (0.47=sigma of the LJ)
                df_Si,lx,ly,lz = Utils.Building.Builder.build_fcc(a,nx,ny,nz)
                df_Si.loc[:,'type'] = bulk_res[0]
                df_Si.loc[:,'bead'] = beads[bulk_res[0]][0]

                # writes gro file of fcc structure
                name = "fcc.gro"
                fout = open(name,"w")
                fout.write(f'System \n {len(df_Si)} \n')
                fout.close()  
                Utils.WandR.WriterAndReader.write_gro(name,"a",df_Si)
            else:
                surf = pd.read_fwf(matrix_name,widths=[5,5,5,5,8,8,8,8,8,8],skiprows=2,skipfooter=1,header=None)[[1,4,5,6]]
                surf.columns = ["res","x","y","z"]
                df_Si = surf[surf["res"]==bulk_res[0]].sort_values(by=["z","y","x"]).reset_index(drop=True)

            # separates final layer and attach silanes to it
            norm_max = np.max(df_Si[sil_norm])
            df_Si_layer = df_Si[df_Si[sil_norm] == norm_max].copy()
            df_Sil, df_layer1, nSils = Silanizer.add_silanes(grafDens,surfDist,nBeads,df_Si_layer,surfGeometry,sil_norm)

            # calculates x and y limits
            xmin, ymin, zmin = np.min(df_Si["x"]), np.min(df_Si["y"]), np.min(df_Si["z"])
            xmax, ymax, zmax = np.max(df_Si["x"]), np.max(df_Si["y"]), np.max(df_Si["z"])

        elif surfGeometry == "cylindrical" or surfGeometry == "slit":
            # reads matrix file and selects the material where there is a pore
            max_a, max_dist = 0.92, 6.34
            pore = pd.read_fwf(matrix_name,widths=[5,5,5,5,8,8,8,8,8,8],skiprows=2,header=None,skipfooter=1)[[1,4,5,6]]
            pore.columns = ["res","x","y","z"]
            df_Si = pore[pore["res"]==bulk_res[0]].sort_values(by=["z","y","x"]).reset_index(drop=True)

            # finds the pore coordinates, sets it as surface layer, and add silanes to it
            df_Si_layer = Silanizer.find_pore(max_a,max_dist,df_Si,surfGeometry)
            df_Sil, df_layer1, nSils = Silanizer.add_silanes(grafDens,surfDist,nBeads,df_Si_layer,surfGeometry,sil_norm)
            # calculates x and y limits
            xmin, ymin, zmin = np.min(df_Si["x"]), np.min(df_Si["y"]), np.min(df_Si["z"])
            xmax, ymax, zmax = np.max(df_Si["x"]), np.max(df_Si["y"]), np.max(df_Si["z"])
            lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin

        # df_layer contains the surface layer beads which are NOT under a molecule
        # the residue name and bead type columns are added to the dataframe
        df_layer = df_Si_layer.drop(labels=df_layer1.index,axis=0,inplace=False)
        df_layer.loc[:,'type'] = surf_res[1]
        df_layer.loc[:,'bead'] = beads[surf_res[1]][0]

        # df_layer1 contains the surface layer beads which are under a molecule
        # the residue name and bead type columns are added to the dataframe
        df_layer1.loc[:,'type'] = surf_res[0]
        df_layer1.loc[:,'bead'] = beads[surf_res[0]][0]

        # df_bulk contains all the beads below surface layer
        # the residue name and bead type columns are added to the dataframe
        df_bulk = df_Si.drop(labels=df_Si_layer.index,axis=0,inplace=False)
        df_bulk.loc[:,'type'] = bulk_res[0]
        df_bulk.loc[:,'bead'] = beads[bulk_res[0]][0]  

        # concatenate dfs
        df = pd.concat([df_Si,df_Sil],ignore_index=True,copy=False)
        df = df.sort_values(by=["z","y","x"], ascending=False).reset_index(drop=True)
        id_all = df.index.array.to_numpy()+1

        #if nBeads > 0 and surfGeometry == "cylindrical":
            # plot_layers plots n layers (first argument) including the border atoms in blue, the border atoms bonded to the silanes in green and the silanes in red, along with the normal vectors
            #Silanizer.plot_layers(3,df_Sil,df_layer,df_layer1,df_Si_layer)

    # when grafting density = 0
    else:   
        if surfGeometry == "flat":
            # creates the fcc structure with given parameters if no surface file is given
            if not matrix_name: 
                nx,ny,nz =50, 10, 9
                a = 0.47*2**(1/6) #lattice space (0.47=sigma of the LJ)
                df_Si,lx,ly,lz = Utils.Building.Builder.build_fcc(a,nx,ny,nz)

            else:
                surf = pd.read_fwf(matrix_name,widths=[5,5,5,5,8,8,8,8,8,8],skiprows=2,skipfooter=1,header=None)[[1,4,5,6]]
                surf.columns = ["res","x","y","z"]
                df_Si = surf[surf["res"]==bulk_res[0]].sort_values(by=["z","y","x"]).reset_index(drop=True)

            id_all = df_Si.index.array.to_numpy()+1
            norm_max = df_Si[sil_norm].max()

            # df_layer contains the surface's first layer beads
            # the residue name and bead type columns are added to the dataframe
            df_layer = df_Si[df_Si[sil_norm] == norm_max].copy()
            df_layer.loc[:,'type'] = surf_res[0]
            df_layer.loc[:,'bead'] = beads[surf_res[0]][0]

            # df_bulk contains all the beads below surface's first layer
            # the residue name and bead type columns are added to the dataframe
            df_bulk = df_Si.drop(labels=df_layer.index,axis=0,inplace=False)
            df_bulk.loc[:,'type'] = bulk_res[0]
            df_bulk.loc[:,'bead'] = beads[bulk_res[0]][0]

            # calculates x and y limits
            xmin, ymin, zmin = np.min(df_Si["x"]), np.min(df_Si["y"]), np.min(df_Si["z"])
            xmax, ymax, zmax = np.max(df_Si["x"]), np.max(df_Si["y"]), np.max(df_Si["z"])
            lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin

        elif surfGeometry == "cylindrical" or surfGeometry == "slit":
            # reads matrix file and selects the material where there is a pore
            max_a, max_dist, fileread = 0.92, 6.34, matrix_name
            pore = pd.read_fwf(fileread,widths=[5,5,5,5,8,8,8,8,8,8],skiprows=2,skipfooter=1,header=None)[[1,4,5,6]]
            pore.columns = ["res","x","y","z"]

            df_Si = pore[pore["res"]==bulk_res[0]][["x","y","z"]].sort_values(by=["z","y","x"]).reset_index(drop=True)

            # finds the pore coordinates and set as surface layer
            # df_layer contains all beads in surface layer
            # the residue name and bead type columns are added to the dataframe
            df_layer = Silanizer.find_pore(max_a,max_dist,df_Si,surfGeometry)
            df_layer.loc[:,'type'] = surf_res[0]
            df_layer.loc[:,'bead'] = beads[surf_res[0]][0]

            # df_bulk contains all the beads below surface's first layer
            # the residue name and bead type columns are added to the dataframe
            df_bulk = df_Si.drop(labels=df_layer.index,axis=0,inplace=False)
            df_bulk.loc[:,'type'] = bulk_res[0]
            df_bulk.loc[:,'bead'] = beads[bulk_res[0]][0]

            # calculates limits
            xmin, ymin, zmin = np.min(df_Si["x"]), np.min(df_Si["y"]), np.min(df_Si["z"])
            xmax, ymax, zmax = np.max(df_Si["x"]), np.max(df_Si["y"]), np.max(df_Si["z"])
            lx, ly, lz = xmax-xmin, ymax-ymin, zmax-zmin

    # initiate variables, array, dfs, to construct the dataframes already in the format for writing the input files for gromacs
    print("Assembling dataframes and writing files...\n")
    bead_names, tp_names = [], []
    df_mol = pd.DataFrame()
    i,j = 0,0

    # creates unified dataframe in correct order to build silane molecules bonded to surface
    if nSils > 0:
        for s in range(nSils):
            k=j
            for ss in range(nBeads):
                df_mol = pd.concat([df_mol,df_Sil.loc[[k]]],axis=0,ignore_index=True)
                bead_names.append(beads[dep_res[0]][ss])
                tp_names.append(dep_res[0]) 
                k+=nSils
            j+=1
            df_mol = pd.concat([df_mol,df_layer1.iloc[[i]]],axis=0,ignore_index=True)
            bead_names.append(beads[surf_res[0]][0])
            tp_names.append(surf_res[0])
            i+=1
        df_mol["bead"], df_mol["type"] = bead_names, tp_names
        df_mol = pd.concat([df_mol,df_layer,df_bulk],axis=0,ignore_index=True)
    else:
        df_mol = pd.concat([df_mol,df_layer,df_bulk],axis=0,ignore_index=True)

    # writes gro file
    name = f"{path}/initial_config.gro"
    fout = open(name,"w")
    fout.close()  
    Utils.WandR.WriterAndReader.write_gro(name,"a",df_mol)

    # writes itp file
    fname =f"{path}/lay1-sil.itp"
    if nBeads > 0 :
        #Silane-Si molecules
        mol_name=dep_res[0]
        fout = open(fname,"w")
        fout.write(f'[ moleculetype ]\n; molname 	nrexcl\n  {mol_name}             1\n\n[ atoms ]\n; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass\n')
        fout.close()
        j=0
        for i in range(1):
            for k in range(len(beads[dep_res[0]])):
                Utils.WandR.WriterAndReader.write_atoms(fname,None,mode="a",id=[j+k+1],atomTp=beads[dep_res[0]][k],rsdNm=dep_res[0],atomNm=beads[dep_res[0]][k],q=0.0,m=72.0)
            Utils.WandR.WriterAndReader.write_atoms(fname,None,mode="a",id=[j+nBeads+1],atomTp=beads[surf_res[0]][0],rsdNm=surf_res[0],atomNm=beads[surf_res[0]][0],q=0.0,m=72.0)
            j += nBeads+1

        fout = open(fname,"a")
        fout.write(f'\n[ bonds ]\n; id1    id2   funType   b0    Kb\n')
        fout.close()

        #Bulk
        j=0
        for i in range(1):
            ids = [j+i+1 for i in range(nBeads)]
            Utils.WandR.WriterAndReader.write_bonds(fname,None,mode="a",id=ids,f=1,l_eq=0.47,k_eq=3800)
            Utils.WandR.WriterAndReader.write_bonds(fname,None,mode="a",id=[nBeads,nBeads+1],f=1,l_eq=0.47,k_eq=3800)
            j += nBeads+1

        if nBeads > 2:
            fout = open(fname,"a")
            fout.write(f'\n[ angles ]\n; i j k 	funct     angle     force.c.\n')
            fout.close()
            Utils.WandR.WriterAndReader.write_angles(fname,None,mode="a",id=[i+1 for i in range(nBeads)],f=2,angle=180.0,k_eq=35.0)
            if nBeads > 3:
                bead = 3
                while bead < nBeads:
                    Utils.WandR.WriterAndReader.write_angles(fname,None,mode="a",id=[bead-1,bead,bead+1],f=2,angle=180.0,k_eq=35.0)
                    bead += 1

        mol_name = "LAY"
        fout = open(fname,"a")
        fout.write(f'\n[ moleculetype ]\n; molname 	nrexcl\n  {mol_name}            1\n\n[ atoms ]\n; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass\n')
        fout.close()
        Utils.WandR.WriterAndReader.write_atoms(fname,None,mode="a",id=[1],atomTp=beads[surf_res[1]][0],rsdNm=surf_res[1],atomNm=beads[surf_res[1]][0],q=0.0,m=72.0)

        mol_name = "BULK"
        fout = open(fname,"a")
        fout.write(f'\n[ moleculetype ]\n; molname 	nrexcl\n  {mol_name}            1\n\n[ atoms ]\n; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass\n')
        fout.close()
        Utils.WandR.WriterAndReader.write_atoms(fname,None,mode="a",id=[1],atomTp=beads[bulk_res[0]][0],rsdNm=bulk_res[0],atomNm=beads[bulk_res[0]][0],q=0.0,m=72.0)

    else:
        mol_name = "LAY"
        fout = open(fname,"w")
        fout.write(f'\n[ moleculetype ]\n; molname 	nrexcl\n  {mol_name}            1\n\n[ atoms ]\n; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass\n')
        fout.close()
        Utils.WandR.riterAndReader.write_atoms(fname,None,mode="a",id=[1],atomTp=beads[surf_res[0]][0],rsdNm=surf_res[0],atomNm=beads[surf_res[0]][0],q=0.0,m=72.0)

        mol_name = "BULK"
        fout = open(fname,"a")
        fout.write(f'\n[ moleculetype ]\n; molname 	nrexcl\n  {mol_name}            1\n\n[ atoms ]\n; id 	type 	resnr 	residu 	atom 	cgnr 	charge    mass\n')
        fout.close()
        Utils.WandR.WriterAndReader.write_atoms(fname,None,mode="a",id=[1],atomTp=beads[bulk_res[0]][0],rsdNm=bulk_res[0],atomNm=beads[bulk_res[0]][0],q=0.0,m=72.0)

    # writes topol file
    if nBeads > 0:
        Utils.WandR.WriterAndReader.write_topol(f"{path}/topol.top","SilSurf",[["SIL",nSils],["LAY",len(df_layer)],["BULK",len(df_bulk)]],['#include "martini_v3.0.0_C1Lay.itp"','#include "martini_v3.0.0_solvents_v1.itp"','#include "lay1-sil.itp"'],"w")
    else:
        Utils.WandR.WriterAndReader.write_topol(f"{path}/topol.top","SilSurf",[["LAY",len(df_layer)],["BULK",len(df_bulk)]],['#include "martini_v3.0.0_C1C2.itp"','#include "lay1-sil.itp"','#include "martini_v3.0.0_solvents_v1.itp"'],"w")

    print("Finished. (:\n")
    
    
def assemble(path):
    blocks=["piston1.gro","initial_config.gro","piston2.gro"]
    positions=[(0,0,0.0),(0,0,24.154),(0,0,65.107)]
    transforms=["none","none","none"]
    dfs_to_concat,delta = [], 0

    print(f"Assembling blocks: {blocks}")
    print(f"Positions: {positions}")
    print(f"Transforms: {transforms}")
    
    Utils.WandR.WriterAndReader.write_topol(f"{path}/topol.top","System",[],['#include "martini_v3.0.0_N1Lay.itp"','#include "martini_v3.0.0_solvents_v1.itp"','#include "lay1-sil.itp"', '#include "surf.itp"'],"w")
    for block,ts,ps in zip(blocks,transforms,positions):
        df_temp = pd.read_fwf(f"{path}/{block}",widths=[5,5,5,5,8,8,8],engine='python',skiprows=2,header=None,skipfooter=1,names=["i1","type","bead","i2","x","y","z"])
        if ts != "none":
            coord = ts[-1]
            df_temp[coord] = df_temp[coord]*(-1)

        xmin, xmax, ymin, ymax, zmin, zmax = np.min(df_temp['x']), np.max(df_temp['x']), np.min(df_temp['y']), np.max(df_temp['y']),np.min(df_temp['z']), np.max(df_temp['z'])
        df_temp["x"], df_temp["y"], df_temp["z"] = df_temp["x"]-xmin, df_temp["y"]-ymin, df_temp["z"]- zmin
        df_temp["x"],df_temp["y"],df_temp["z"] = df_temp["x"]+ps[0],df_temp["y"]+ps[1],df_temp["z"]+ps[2] 

        tops = {}
        for tp in df_temp["type"].unique():
            if tp == "SIL" or tp == "SI":
                tops["SIL"] = len(df_temp[ (df_temp["type"]=="SIL") & (df_temp["bead"]=="C2") ] )
            else:
                tops[tp] = len(df_temp[df_temp["type"]==tp])

        list = [[tp,tops[tp]] for tp in [c for c in df_temp["type"].unique() if c != "SI"]]
        Utils.WandR.WriterAndReader.write_topol(f"{path}/topol.top","",list,[],"a")
        dfs_to_concat.append(df_temp)

    df_all = pd.concat(dfs_to_concat,axis=0,ignore_index=True)
    Utils.WandR.WriterAndReader.write_gro(f"{path}/solid.gro","w",df_all)

    print("\nFinished. (:\n")
    
    
def solvate(path,ngas,mol): 
    os.system(f"cd {path}; gmx_mpi insert-molecules -ci {mol}.gro -nmol {ngas} -o {mol}_box_left.gro -try 100 -box 20 20 20 >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi insert-molecules -ci {mol}.gro -nmol {ngas} -o {mol}_box_right.gro -try 100 -box 20 20 20 >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi solvate -cs wbox.gro -cp {mol}_box_left.gro -scale 1 -box 20 20 20 -o left_box.gro >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi solvate -cs wbox.gro -cp {mol}_box_right.gro -scale 1 -box 20 20 20 -o right_box.gro >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi editconf -f left_box.gro -center 10 10 13 -o left_box.gro -box 20.04700000 19.95000000 67.26100000 >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi editconf -f right_box.gro -center 10 10 54 -o right_box.gro -box 20.04700000 19.95000000 67.26100000 >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi solvate -cs right_box.gro -cp left_box.gro -box 20.04700000 19.95000000 67.26100000 -o wboxes.gro >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi solvate -cs wboxes.gro -cp solid.gro -scale 1 -box 20.04700000 19.95000000 67.26100000 -o start.gro -p topol.top >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi editconf -f start.gro -c -box 20.04700000 19.95000000 200 -o start.gro >/dev/null 2>&1")
    
def minimize(root,path,sys,mol):
    os.system(f"cd {path}; cp {root}/scripts/run_min* .; cp {root}/scripts/*.itp . >/dev/null 2>&1")
    os.system(f"cd {path}; gmx_mpi grompp -f run_min_{mol}.mdp -c start.gro -p topol.top -maxwarn 1 >/dev/null 2>&1")
    os.system(f"cd {path}; sbatch --job-name={sys} run_min.sh")
     
def piston_sim(root,path,sys,mol,press):
    os.system(f"cd {path}; mkdir {press}MPa; cp min/*.itp {press}MPa/; cp min/topol.top {press}MPa/{press}MPa.top; cp min/minimized.gro {press}MPa/start.gro; cp {root}/scripts/start_files/run_pull_cg_{mol}.mdp {press}MPa/; cp {root}/scripts/start_files/run_leo.sh {press}MPa/; cp {root}/scripts/start_files/press_dimezzata.py {press}MPa/")
    os.system(f'cd {path}/{press}MPa/; sed -i "s/PRESSURE/"{press}"/g" press_dimezzata.py; force=$(python3 press_dimezzata.py); echo $force; sed -i "s/SEDFORCE/"$force"/g" run_pull_cg_{mol}.mdp')
    
    os.system(f"cd {path}/{press}MPa/; gmx_mpi grompp -f run_pull_cg_{mol}.mdp -c start.gro -p topol.top -o {press}MPa.tpr -maxwarn 1 >/dev/null 2>&1")
    os.system(f"cd {path}/{press}MPa/; sbatch --job-name={sys}_{press}MPa run_leo.sh {press}MPa")
    
def extrusion(root,path,sys,mol,press):
    os.system(f"cd {path}; mkdir extrusion; cp {root}/scripts/start_files/extrusion_float_clust.sh extrusion/; cp {root}/scripts/start_files/run_pull_cg_{mol}.mdp extrusion/; cp {root}/scripts/start_files/run_leo.sh extrusion/; cp {root}/scripts/start_files/press_dimezzata.py extrusion/")
    os.system(f"cd {path}/extrusion; sbatch --job-name=extrusion_sender extrusion_float_clust.sh {press} 1 30 {path} {mol} {sys}")
    
def intrusion(root,path,sys,mol,press):
    os.system(f"cd {path}; mkdir intrusion; cp {root}/scripts/start_files/intrusion_float_clust.sh intrusion/; cp {root}/scripts/start_files/run_pull_cg_{mol}.mdp intrusion/; cp {root}/scripts/start_files/run_leo.sh intrusion/; cp {root}/scripts/start_files/press_dimezzata.py intrusion/")
    os.system(f"cd {path}/intrusion; sbatch --job-name=intrusion_sender intrusion_float_clust.sh {press} 1 30 {path} {mol} {sys}")