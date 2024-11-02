
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from Silanizer.Grafter.grafter import Assembling as ass
from Silanizer.Grafter.utils import WandR as wr
from Silanizer.Grafter.utils import Building as bld
from Silanizer.Grafter.analysis import Plotting as pltt
from Silanizer.Grafter.analysis import RunningMSD as rmsd
from Silanizer.Grafter.analysis import RunningLH as lh
from Silanizer.Grafter.analysis import RunningDEN as rden
from Silanizer.Grafter.analysis import RunningCA as ca
from Silanizer.Grafter.analysis import RunningSURF as surf
from collections import Counter
import MDAnalysis as mda
import subprocess       
from tqdm.notebook import tqdm
from scipy.stats import gamma
import random
from natsort import natsorted

#defining a class called NewSystem
#inherits methods of all auxiliary classes

class NewSystem(ass.Assembler, bld.Builder, wr.WriterAndReader, pltt.Plotter, rmsd.MSD, lh.LayerHeight, rden.DensityProfile, ca.ContactAngle, surf.Surface):
    """
    A class representing a new system for grafting.

    Attributes:
    - maxA: The maximum value for A.
    - maxDist: The maximum distance.
    - folder: The root folder.
    - name: The name of the system.
    - dfs: A dictionary of dataframes.
    - dictNames: A dictionary of names.
    - surfDist: The surface distance.
    - grafDens: The grafting density.
    - matrix: The matrix information.
    - dispersity: The chain dispersity information.
    - surfGeometry: The surface geometry.
    - tiltMol: The tilt molecule information.
    - dimensions: The dimensions of the system.
    - uBulk: The universe object for the bulk.
    - universe: The universe object for the system.

    Methods:
    - read_inputs: Reads the inputs from a file or dictionary.
    - update_names: Updates the names of the dataframes.
    - update_dimensions: Updates the dimensions of the system.
    - make_hole: Makes a hole in the dataframe.
    - universe_from_df: Creates a universe object from a dataframe.
    - build_surface: Builds the surface of the system.
    - graft_matrix: Grafts the matrix onto the surface.
    - unify_df: Creates a unified dataframe.
    - make_itps: Generates the itp files.
    """
    
    maxA, maxDist = 0.92, 6.34

    def __init__(self, root=None, gro=None, traj=None, name="System", molSizes=[]):
        """
        Initializes a new NewSystem object.

        Parameters:
        - root: The root folder.
        - gro: The gro file.
        - name: The name of the system.
        """
        self.folder = root
        self.name = name
        self.dfs = {}
        self.dictNames = {}
        self.molSizes = molSizes

        if gro:
            pathGro = gro
            
            if traj:
                pathTraj = traj
                u = mda.Universe(pathGro,pathTraj)
                self.universe = u          
            else:
                u = mda.Universe(pathGro)
                self.universe = u

    def plot_mol_distribution(self, ax=None):
        if ax is None:
            fig,ax = plt.subplots()
        ax.hist(self.molSizes, bins=100, alpha=0.5, color="xkcd:tomato", histtype="stepfilled", label="Molecule size distribution")
        ax.set_xlabel("N")
        ax.set_ylabel("Count")
        return ax
    
    def unwrap_coordinates(self, positions, box):
        # Calculate the unwrapped positions
        unwrapped_positions = np.zeros_like(positions)
        unwrapped_positions[0] = positions[0]  # Assume the first position is already unwrapped

        for i in range(1, positions.shape[0]):
            delta = positions[i] - positions[i - 1]
            delta -= np.round(delta / box) * box
            unwrapped_positions[i] = unwrapped_positions[i - 1] + delta

        return unwrapped_positions
            
    def update_names(self, dfnames, in_dict_names):
        """
        Updates the names of the dataframes.

        Parameters:
        - dfnames: The list of dataframe names.
        - in_dict_names: The dictionary of input names.
        """
        for key in dfnames:
            self.dfs[key]["type"] = in_dict_names[key][0]
            self.dfs[key]["bead"] = in_dict_names[key][1]
            if key not in self.dictNames:
                self.dictNames[key] = in_dict_names[key]

    def get_mol_sizes(self, molname, outfile=None):
        import re
        u = self.universe
        sel = u.select_atoms(f"name {molname} or resname {molname}")
        resnames = sel.residues.resnames
        molSizes = natsorted(list(set([int(re.findall(r'\d+', resname)[0]) for resname in resnames])))

        if outfile is not None:
            out = open(self.folder+"/"+outfile, "w")
            for m in molSizes:
                out.write(f"{m}\n")
            out.close()
        return molSizes

    def update_dimensions(self, df):
        """
        Updates the dimensions of the system.

        Parameters:
        - df: The dataframe.
        """
        lx, ly, lz = df["x"].max()-df["x"].min(), df["y"].max()-df["y"].min(), df["z"].max()-df["z"].min()
        self.dimensions = [lx,ly,lz]

    def make_hole(self, df, pore_radius):
        """
        Makes a hole in the dataframe.

        Parameters:
        - df: The dataframe.
        - pore_radius: The pore radius.

        Returns:
        - The modified dataframe.
        """
        xc = df["x"].mean()
        yc = df["y"].mean()
        df_new = df.where(( (df["x"]-xc)**2+(df["y"]-yc)**2 >= pore_radius**2 )).dropna()
        maxz = df["z"].max()
        df_new = df_new.where(  ( (df_new["x"]-xc)**2+(df_new["y"]-yc)**2 < (pore_radius+0.4)**2 ) & (df_new["z"]>1) & (df_new["z"]<maxz-1 ) | ( (df_new["z"]<1) | (df_new["z"]>maxz-1) )   ).dropna().reset_index(drop=True).round(decimals=3)
        
        return df_new

    def build_surface(self, names=None, radius=None, matrix=None):
        """
        Builds the surface of the system.

        Parameters:
        - names: The names of the surface.
        - radius: The radius of the surface.
        - matrix: The matrix information.
        """
        if not names:
            names = self.dictNames["bulk"]
        if not matrix:
            matrix = self.matrix
        else:
            self.matrix = matrix

        name_atom, name_residue = names
        if self.matrix[0] == "build":
            nx,ny,nz = self.matrix[1]
            a = 0.47*2**(1/6)
            df_bulk,lx,ly,lz = self.build_fcc(a,nx,ny,nz)
            self.dfs["bulk"] = df_bulk

        elif self.matrix[0] == "file":
            u = mda.Universe("file")
            atoms = u.select_atoms("resname "+name_residue)
            self.dfs["bulk"] = pd.DataFrame(atoms.positions*.1).sort_values(by=["z","y","x"]).reset_index(drop=True)

            if self.surfGeometry == "cylindrical" and radius:
                self.dfs["bulk"] = self.make_hole(self.dfs["bulk"],radius)

        self.update_names(["bulk"], {"bulk": [name_residue, name_atom]})   
        self.update_dimensions(self.dfs["bulk"])
        self.uBulk = self.universe_from_df(self.dfs["bulk"], box=self.dimensions, convert_unit=10)

        try:
            self.universe = mda.Merge(self.universe,self.uBulk)
        except AttributeError:
            self.universe = self.uBulk

    def graft_matrix(self, tiltMol=0, surfNorm="z"):
        """
        Grafts the matrix onto the surface.

        Parameters:
        - tiltMol: The tilt molecule information.
        - surfNorm: The surface normal direction.
        """
        in_dict_names = self.dictNames.copy()

        if self.dispersity[0] == "poly" and self.dispersity[1]:
            args = self.dispersity[1]
            print("\nUsing Schultz-zimm distribution", "with args", args)
            chainSize = None
            self.distParams = args

        elif self.dispersity[0] == "mono" and self.dispersity[1]:
            chainSize = self.dispersity[1]
            print("\nUsing chain size", chainSize)
            args = None
            self.chainSize = chainSize

        else:
            raise Exception("No chain size or distribution given, please provide one")


        if self.surfGeometry == "flat":
            norm_max = np.max(self.dfs["bulk"][surfNorm])
            self.dfs["layer"] = self.dfs["bulk"][self.dfs["bulk"][surfNorm] == norm_max].copy()

        elif self.surfGeometry == "cylindrical" or self.surfGeometry == "slit":
            self.dfs["layer"] = find_pore(self.maxA, self.maxDist, self.dfs["bulk"], self.surfGeometry)

        # attaches silanes to the surface layer
        self.dfs["polymer"], self.dfs["under_polymer"], self.nSils, self.molSizes = add_silanes(self.grafDens, self.surfDist, self.dfs["layer"], self.surfGeometry, surfNorm, tiltMol=tiltMol,fromDistribution=args, chain_size=chainSize)
        self.dfs["bulk"] = self.dfs["bulk"].drop(labels=self.dfs["layer"].index, axis=0, inplace=False)
        self.dfs["layer"] = self.dfs["layer"].drop(labels=self.dfs["under_polymer"].index, axis=0, inplace=False)

        in_dict_names["under_polymer"][0] = in_dict_names["under_polymer"][0][0]
        self.update_names(self.dfs.keys(), in_dict_names)
    
        self.unify_df()
        self.universe = self.universe_from_df(self.dfs["unified"], box=self.dimensions, convert_unit=10)

    def unify_df(self):
        """
        Creates a unified dataframe.
        """
        df_mol = pd.DataFrame({"x":[],"y":[],"z":[],"bead":[],"type":[]})

        j,jj,k = 0,0,0
        if self.nSils > 0:
            for s in range(self.nSils):
                kk = k + self.molSizes[s]
                jj = j + self.molSizes[s]
                
                df_mol = pd.concat([df_mol,self.dfs["polymer"].iloc[k:kk]],axis=0,ignore_index=True)
                df_mol.iloc[j:j+1, df_mol.columns.get_loc('type')] = self.dictNames["polymer"][0][0]+str(self.molSizes[s])
                df_mol.iloc[j:j+1, df_mol.columns.get_loc('bead')] = "END"
                df_mol.iloc[j+1:jj, df_mol.columns.get_loc('type')] = self.dictNames["polymer"][0][0]+str(self.molSizes[s])
                df_mol.iloc[j+1:jj, df_mol.columns.get_loc('bead')] = self.dictNames["polymer"][1]

                df_mol = pd.concat([df_mol,self.dfs["under_polymer"].iloc[[s]]],axis=0,ignore_index=True)
                df_mol.iloc[-1,df_mol.columns.get_loc('type')] = df_mol.iloc[-1, df_mol.columns.get_loc('type')]+str(self.molSizes[s])

                k = kk
                j = jj + 1
                
        df_unified = pd.concat([df_mol, self.dfs["layer"], self.dfs["bulk"]], axis=0, ignore_index=True)
        self.dfs["unified"] = df_unified
        self.update_dimensions(df_unified)


    def make_itps(self, rep_range, dict_names=None):
        """
        Generates the itp files.

        Parameters:
        - rep_range: The range of repetitions.
        """

        if dict_names is None:
            if not self.dictNames:
                raise Exception("No dictionary of names given, please provide one")
        else:
            self.dictNames = dict_names
             

        def generate_itp(nrep):
            out_file = open(f"{self.folder}/itps/lay_pdms_{nrep}.itp","w")

            #molecule type
            out_file.write(f"[ moleculetype ]\n; molname  nrexcl\n  {self.dictNames['polymer'][0]}{nrep}       1\n\n[ atoms ]\n")
            for i in range(1,nrep+1):
                if i == 1 :
                    out_file.write(f"{i} {self.dictNames['polymer'][1]}   {i} {self.dictNames['polymer'][0][0]}{nrep}  END   {i} 0.0 72.0\n")
                else:
                    out_file.write(f"{i} {self.dictNames['polymer'][1]}   {i} {self.dictNames['polymer'][0][0]}{nrep}  {self.dictNames['polymer'][1]}   {i} 0.0 72.0\n")

            tip = nrep+1
            out_file.write(f"{tip} {self.dictNames['under_polymer'][1]} {tip} {self.dictNames['under_polymer'][0][0]}{nrep} {self.dictNames['under_polymer'][1]} {tip} 0.0 72.0\n")

            #bonds
            out_file.write("\n[ bonds ]\n")
            for i in range(2,nrep):
                j = i+1 
                out_file.write(f"  {i}   {j} 1 0.448 11500\n")
            out_file.write("\n; END-PDMS\n")
            out_file.write(" 1   2 1 0.446 11000\n")
            out_file.write("\n; PDMS-SURF\n")
            out_file.write(f" {nrep}   {tip} 1 0.470 3800\n")

            #angles
            if nrep > 2:
                out_file.write("\n[ angles ]\n")
                for i in range(2,nrep-2):
                    j = i+1
                    k = i+2
                    out_file.write(f"  {i}   {j}   {k} 1 86 45.89\n")
                
            
                out_file.write("\n; END-PDMS\n")
                out_file.write("  1   2   3 1 87 78\n")
                out_file.write("\n; PDMS-SURF\n")
                pre_tip = nrep-1 
                out_file.write(f" {pre_tip}   {nrep}   {tip} 1 180 35\n")

            #dihedrals
            if nrep > 3:
                out_file.write("\n[ dihedrals ]\n")
                for i in range(2,nrep-3):
                    j = i+1
                    k = i+2
                    l = i+3
                    out_file.write(f"  {i}   {j}   {k}   {l}  1 1.18 1.4 1\n")
                out_file.write("\n; END-PDMS\n")
                out_file.write("  1   2   3   4 1 1.85 1.2 1\n")
                out_file.write("\n; PDMS-SURF\n")
                #out_file.write(f"  {i+1}   {j+1}   {k+1}   {l+1}  1 1.18 1.4 1")
            out_file.close()

        subprocess.run(f"mkdir -p {self.folder}/itps;", shell=True, executable="/bin/bash")

        print("Generating itp files")
        for nrep in tqdm(rep_range):
            generate_itp(nrep)


    def out_topology(self, fname="initial_config.gro", out_sizes=True):
        name = f"{self.folder}/{fname}"
        self.universe.atoms.positions = np.array(self.universe.atoms.positions)
        self.universe.dimensions[:2] = self.universe.dimensions[:2]
        self.universe.dimensions[2] = self.universe.dimensions[2]+200
        gro_writer = mda.coordinates.GRO.GROWriter(name, n_atoms=len(self.universe.atoms))
        gro_writer.write(self.universe)
        gro_writer.close()

        if out_sizes:
            out = open(f"{self.folder}/molSizes.dat", "w")
            for m in self.molSizes:
                out.write(f"{m}\n")
            out.close()

        PolySizes = [f"{self.dictNames['polymer'][0]}{n}"for n in self.molSizes] 
        PolyCounts = Counter(PolySizes)
        PolyMols = [[p,1] for p in PolySizes]
        SurfMols = [[self.dictNames["layer"][0], len(self.dfs["layer"])], [self.dictNames["bulk"][0], len(self.dfs["bulk"])]]
        includes = ['#include "itps/DMS_martini3_v1.itp"', '#include "itps/martini_v3.0.0_solvents_v1.itp"', '#include "itps/surf.itp"','#include "itps/martini_v3.0.0_small_molecules_v1.itp"', '"#include "itps/END.martini3.ff"'] + [f'#include "itps/lay_pdms_{n.split("PDMS")[1]}.itp"' for n in PolyCounts]
        name = f"{self.folder}/topol.top"
        self.write_topol(name,"PDMS_brush",PolyMols+SurfMols,includes,"w")


def cylinderFitting(x,y,z,p,axis,plot="no"): # function to fit a cylindrical equation to data
    """
    p is initial values of the parameter;
    p[0] = Xc, x coordinate of the cylinder centre
    P[1] = Yc, y coordinate of the cylinder centre
    P[2] = alpha, rotation angle (radian) about the x-axis
    P[3] = beta, rotation angle (radian) about the y-axis
    P[4] = r, radius of the cylinder
    """   
    def plot_cylinder_fit(x,y,z,est_p,minh,maxh): # function to plot cylindrical fit
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter(x,y,z, alpha=0.3, c="black")
        z1 = np.linspace(minh, maxh, 50)
        theta = np.linspace(0, 2*np.pi, 50)
        theta_grid, z_grid=np.meshgrid(theta, z1)
        x_grid = est_p[2]*np.cos(theta_grid) + est_p[0]
        y_grid = est_p[2]*np.sin(theta_grid) + est_p[1]
        ax.plot_surface(x_grid, y_grid, z_grid, cmap=plt.cm.YlGnBu_r)
        plt.savefig("cylinder_fit.png")
        
        fig1 = plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.scatter(x,y,z, alpha=0.3, c="black")
        ax1.view_init(90,90)
        ax1.plot_surface(x_grid, y_grid, z_grid, cmap=plt.cm.YlGnBu_r)
        plt.savefig("cylinder_fit_top.png")
    
    def compute_cylinder_area(r,h): # function to calculate the surface area of the fitted cylinder
        h = float(h)
        pi = 3.14159
        surface_area = 2 * pi * r ** 2 + 2 * pi * r * h
        return surface_area

    # the "axis" choice is to check if the fitted cylinder can be tilted in relation to the z axis. If "vertical" it means is fixed in the z axis
    if axis == "vertical":
        p = [p[0],p[1],p[4]]
        fitfunc = lambda p, x, y, z: (- np.cos(0.0)*(p[0] - x) - z*np.cos(0.0)*np.sin(0.0) - np.sin(0.0)*np.sin(0.0)*(p[1] - y))**2 + (z*np.sin(0.0) - np.cos(0.0)*(p[1] - y))**2 #fit function
        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[2]**2 #error function 
    else:
        fitfunc = lambda p, x, y, z: (- np.cos(p[3])*(p[0] - x) - z*np.cos(p[2])*np.sin(p[3]) - np.sin(p[2])*np.sin(p[3])*(p[1] - y))**2 + (z*np.sin(p[2]) - np.cos(p[2])*(p[1] - y))**2 #fit function
        errfunc = lambda p, x, y, z: fitfunc(p, x, y, z) - p[4]**2 #error function 

    est_p , success = leastsq(errfunc, p, args=(x, y, z), maxfev=1000)
    if plot=="yes":
        plot_cylinder_fit(x,y,z,est_p,np.min(z),np.max(z))

    return est_p, compute_cylinder_area(est_p[2],np.max(z)-np.min(z))

def add_silanes(grafDens,surfDist,df_cut,surfGeo,sil_norm,tiltMol=0,fromDistribution=None,chain_size=None,holes=None):  
    """
    Function to add silanes to a surface.
    Input parameters
        - grafDens : grafting density of molecules
        - surfDist : distance in which the molecules will be placed above the surface. It is also the distance between the beads of the molecule
        - df_cut : the dataframe containing the surface coordinates (and normals to orient molecules in case of cylindrical surface)
        - surfGeo : the geometry of the surface, which can be "flat" or "cylindrical"
    Output
        - df_Sil : dataframe containing coordinates of deposited molecules
        - df_layer1 : dataframe containing the coordinates of the surface beads which are bonded to a deposited molecule
        - numOfSilanes : number of deposited molecules
        - fromDistribution : give a function from which to sample the number of beads in the molecule
    """   

    if surfGeo == "flat":
        # calculates area of grafting, maximum grafting density, and number of silanes to graft from given density
        coords = ["x","y","z"]
        axis = [i for i in coords if i != sil_norm]
        xi, yi = df_cut[axis[0]], df_cut[axis[1]]

        sideX, sideY = np.max(xi)-np.min(xi), np.max(yi)-np.min(yi)
        areaXY = sideX*sideY
        maxGrafDens, numOfSilanes = float(len(df_cut)/areaXY), round(grafDens*areaXY)
        

    elif surfGeo == "cylindrical" or "slit":
        # calculates area of grafting, maximum grafting density and number of silanes to graft from given density
        p = np.array([np.mean(df_cut["x"]),np.mean(df_cut["y"]),0,0,1.0])
    
        if surfGeo=="cylindrical":
            est_p, areaXY =  cylinderFitting(df_cut["x"],df_cut["y"],df_cut["z"],p,"vertical",plot="yes")
        
        elif surfGeo=="slit":
            areaXY = (df_cut["z"].max()-df_cut["z"].min())*(df_cut["x"].max()-df_cut["x"].min())*2
        
    maxGrafDens, numOfSilanes= float(len(df_cut)/areaXY), round(grafDens*areaXY)
    if grafDens > maxGrafDens:
        raise ValueError("Maximum grafting   density is %.3f!\n" % (maxGrafDens)) 

    # picks random list of surface beads
    idx = np.random.choice(df_cut.index, numOfSilanes, replace=False)

    # place molecules above selected surface beads by a distance equal to surfDist. Molecule beads are also separated by surfDist
    xS,yS,zS = [],[],[]
    ii = 0
    molSizes = []
    
    last_pos = [0,0,0]
    for k in idx:
        molSize = 0
        df_calc = df_cut.loc[k]

        if fromDistribution:
            a = 1/(fromDistribution[0]-1)
            b = a
            while molSize == 0:
                molSize = int(gamma.rvs(a, scale=fromDistribution[1]/b, size=1)[0])
        elif chain_size:
            molSize = chain_size
        else:
            "No distribution or chain size given!"
            return

        molSizes.append(molSize)
        for j in (range(molSize,0,-1)):

            if holes:
                xp, yp, zp = df_calc["x"] + surfDist * df_calc["normx"] * j, df_calc["y"] + surfDist * df_calc["normy"] * j, df_calc["z"]
            else:
                xp, yp, zp = df_calc[axis[0]] + last_pos[0] * random.uniform(0,tiltMol), df_calc[axis[1]] + last_pos[1] * random.uniform(0,tiltMol), df_calc[sil_norm] + j * surfDist + last_pos[2] * random.uniform(0,tiltMol)

            last_pos = xp, yp, zp

            xS.append(xp); yS.append(yp); zS.append(zp)

        ii = ii+1
        sys.stdout.write("Adding molecules: %.2f%%   \r" % (float(ii)/float(numOfSilanes)*100) )
        sys.stdout.flush()   

    print("\nNumber of molecules: %d  -  Number of spots: %d  -  Max. grafting dens.: %.3f\n" % (numOfSilanes,len(df_cut),maxGrafDens))

    # assemble dataframe from molecule and surface bead coordinates
    if holes:
        df_Sil = pd.DataFrame({"x":xS,"y":yS,"z":zS})
    else:
        df_Sil = pd.DataFrame({axis[0]:xS, axis[1]:yS, sil_norm:zS})

    df_lay1 = df_cut.loc[idx]

    return df_Sil, df_lay1, numOfSilanes, molSizes

def find_pore(a,max_dist,cyl,geometry): # function to find the coordinates of a cylindrical pore
    """
    Function to find the coordinates of a cylindrical pore.
    Input parameters
        - a : maximum distance between beads in the ordered structure
        - max_dist : maximum distance allowed by user in which two surface beads can be apart
        - cyl : dataframe containing coordinates of the material in which is the cylindrical pore
    Output
        - df: dataframe with pore coordinates
    """   

    def find_normals(df,zlay,geo): # function to find the normal vectors to then points in which a cylinder function is fitted
        def find_centroid(df): # function to find the centroid of 2d collection of points
            xc, yc = np.mean(df["x"]),np.mean(df["y"])
            return xc, yc

        # for each z layer, calculates the normal vector of each bead 
        normx, normy = [], []
        for zi in zlay:
            lay = df[df["z"]==zi]
            ctd = find_centroid(lay)
            for x,y in zip(lay["x"],lay["y"]):
                if geo=="slit" or "flat":
                    norm = np.sqrt((ctd[1]-y)**2 + (x-x)**2)
                    normx.append(x/norm-x/norm)
                    normy.append(ctd[1]/norm-y/norm) 
                elif geo=="cylindrical":
                    norm = np.sqrt((ctd[1]-y)**2 + (ctd[0]-x)**2)
                    normx.append(ctd[0]/norm-x/norm)
                    normy.append(ctd[1]/norm-y/norm) 
        return normx,normy

    def find_hole(a,x,y,z,coords,max_dist,ax): # function to find a hole in a line of beads. Returns True if finds a hole and False if not
        # if ax=0, searches holes in x direction for a constant y
        tst=False
        if ax == 0:
            x = x.sort_values()
            idxs = x.index

            # walks the beads and append to coords the coordinates and index of the ones in which the distance is bigger than a and smaller than max_dist, that is, the borders of a "hole"
            for i in range(len(idxs)-1):
                p1, p2 = x.loc[idxs[i]], x.loc[idxs[i+1]]
                dx = p2 - p1
                if dx > a and dx < max_dist:
                    tst=True
                    if [idxs[i],p1,y,z] not in coords:
                        coords.append([idxs[i],p1,y,z])
                    if [idxs[i+1],p2,y,z] not in coords:
                        coords.append([idxs[i+1],p2,y,z])
        # if ax=1, searches holes in y direction for a constant x
        elif ax == 1:
            y = y.sort_values()
            idxs = y.index

            # walks the beads and append to coords the coordinates and index of the ones in which the distance is bigger than a and smaller than max_dist, that is, the borders of a "hole"
            for i in range(len(idxs)-1):
                p1, p2 = y.loc[idxs[i]], y.loc[idxs[i+1]]
                dy = p2 - p1
                if dy > a  and dy < max_dist:
                    tst=True
                    if [idxs[i],p1,y,z] not in coords:
                        coords.append([idxs[i],x,p1,z])
                    if [idxs[i+1],p2,y,z] not in coords:
                        coords.append([idxs[i+1],x,p2,z])
        else:
            print("Axis must be 0 or 1")
        return tst

    # for each z layer, finds holes in x and y direction, starting from the center
    z_layers = sorted(set(cyl["z"]))
    ii, coords = 0, []

    # loop throug z_layers which are sorted from lower to higher
    print("\n")
    for zi in z_layers:
        xy = cyl[cyl["z"]==zi][["x","y"]] # gets only the current z layer zi
        x_layers, y_layers = sorted(set(xy["x"])), sorted(set(xy["y"])) # gets unique values of x and y and sorts them from lower to higher to get an ordered "line" of beads
        lx, ly, lz = len(x_layers), len(y_layers), len(z_layers)

        leftx, lefty = int(lx/2), int(ly/2); rightx, righty = leftx, lefty # here we define that half of the length of the x and y layers will be approximately the coordinate of the center of the hole. Of course, it is only true if the pore is really centered in x and y.

        # Test to stop searching for holes. t1, t2, t3 and t4 are False or True values returned by the find_hole() function. When True, a hole is found and the border atoms are added to coords list. When False, nothing happens. Each True or False value is added to a list and only if the last 10 tests have given False the tstx and tsty can change to False. From the middle, it goes to the next "line" in the left and in the right so it can test both in the same instance of the loop.
        while (lefty > 0 or righty < ly or leftx > 0 or rightx < lx):
            if lefty > 0:
                lefty = lefty-1
                yl = xy[xy["y"] == y_layers[lefty]]
                t1 = find_hole(a,yl["x"],y_layers[lefty],zi,coords,max_dist,0)          
            if righty < ly:
                yr = xy[xy["y"] == y_layers[righty]]
                t2 = find_hole(a,yr["x"],y_layers[righty],zi,coords,max_dist,0)
                righty = righty+1

            if leftx > 0:
                leftx = leftx-1
                xl = xy[xy["x"] == x_layers[leftx]]
                t3 = find_hole(a,x_layers[leftx],xl["y"],zi,coords,max_dist,1)
            if rightx < lx:
                xr = xy[xy["x"] == x_layers[rightx]]
                t4 = find_hole(a,x_layers[rightx],xr["y"],zi,coords,max_dist,1)
                rightx = rightx+1

        ii = ii+1
        sys.stdout.write("Finding pore: %.2f%%   \r" % (float(ii)/float(lz)*100) )
        sys.stdout.flush()

    print("\n")
    # assemble dataframe froom coords and add the normals to it. It also makes sure to remove any eventual duplicated coordinates.
    df = pd.DataFrame(coords,columns=["index","x","y","z"])
    df["normx"], df["normy"] = find_normals(df,z_layers,geometry)
    df.drop_duplicates(subset=None, keep='first', inplace=True, ignore_index=True)
    df = df.set_index("index").rename_axis(None)
    return df

def plot_layers(n,df_Sil,df_layer,df_layer1,df_Si_layer): # simple function to plot n layers of the system
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_Sil["x"],df_Sil["y"],df_Sil["z"])
    ax.scatter(df_Si_layer["x"],df_Si_layer["y"],df_Si_layer["z"])
    ax.scatter(df_layer1["x"],df_layer1["y"],df_layer1["z"])
    plt.savefig("cylinder_silane.png")

    zlays1 = sorted(set(df_Sil["z"]))
    for z1 in zlays1[:n]:
        fig1 = plt.figure(figsize=(10, 10))
        ax1 = fig1.add_subplot(111)
        zlay = df_layer1[df_layer1["z"]==z1]
        zlay1 = df_Sil[df_Sil["z"]==z1]
        zlay2 = df_layer[df_layer["z"]==z1]

        ax1.scatter(zlay1["x"],zlay1["y"],c="red")
        ax1.scatter(zlay["x"],zlay["y"],c="blue")
        ax1.scatter(zlay2["x"],zlay2["y"],c="green")

        for x,y,nx,ny in zip(zlay["x"],zlay["y"],zlay["normx"],zlay["normy"]):
            ax1.quiver(x, y, nx, ny)
        plt.savefig(f"normals_{z1}.png")  

# def find_closest(df1,df2):
#     ind = []
#     for _, row1 in df1.iterrows():
#         distSqr = []
#         for index2, row2 in df2.iterrows():
#             d2 = (row2["x"]-row1["x"])**2 + (row2["y"]-row1["y"])**2 + (row2["z"]-row1["z"])**2
#             distSqr.append([index2,d2])
#         distSqr.sort(key=lambda x: x[1])
#         ind.append(distSqr[0][0])
#     return ind

def load_inputs(path):
    import json
    import subprocess
    inputs = json.load(open(f"{path}","r"))

    sample = inputs["name"]
    folder = inputs["folder"]
    surfDist = inputs["surface_distance"]
    grafDens = inputs["grafting_density"]
    matrix = ["file", inputs["matrix"]["file"]] if inputs["matrix"]["file"] else ["build", inputs["matrix"]["size"]]
    dispersity = ["mono", inputs["chain dispersity"]["monodisperse"]] if inputs["chain dispersity"]["monodisperse"] else ["poly", inputs["chain dispersity"]["polydisperse"]]
    surfGeometry = "cylindrical" if inputs["surface geometry"]["cylindrical"] else "flat"
    dictNames = inputs["atom names"]
    tiltMol = inputs["perturbation"]

    print(f"name: {sample}\nfolder: {folder}\nsurface distance: {surfDist}\ngrafting density: {grafDens}\nmatrix: {matrix}\ndispersity: {dispersity}\nsurface geometry: {surfGeometry}\natom names: {dictNames}\ntilt molecule: {tiltMol}\n")

    subprocess.run(f"mkdir -p {folder};",
                shell=True, executable="/bin/bash")
    print(" ")

    return folder, surfDist, grafDens, matrix, dispersity, surfGeometry, dictNames, tiltMol, sample

