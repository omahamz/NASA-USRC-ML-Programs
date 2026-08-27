# -*- coding: mbcs -*-
import os
import time
from part import *
from material import *
from section import *
from assembly import *
from step import *
from interaction import *
from load import *
from mesh import *
from optimization import *
from job import *
from sketch import *
from visualization import *
from connectorBehavior import *

# ============================================================
# CONFIGURATION
# ============================================================

USERNAME = "Pablo"

SCRIPT_DIR = rf"C:\{USERNAME}\PythonScripting\PipelineWorkD"
OUTPUT_DIR = rf"C:\{USERNAME}\PythonScripting\FDData"
STP_DIR    = rf"C:\{USERNAME}\PythonScripting\StepFiles"

STP_FILES = [f for f in os.listdir(STP_DIR) if f.lower().endswith(('.stp', '.step'))]

if not STP_FILES:
    raise RuntimeError('No .stp/.step files found in: {}'.format(STP_DIR))

MAX_CPUS      = 16
CPUS_PER_JOB  = 2
MAX_PARALLEL  = MAX_CPUS // CPUS_PER_JOB
POLL_INTERVAL = 30

# ============================================================
# PARALLEL JOB MANAGER
# ============================================================

def job_is_complete(job_name, job_dir):
    sta_path = os.path.join(job_dir, job_name + '.sta')
    if not os.path.exists(sta_path):
        return False
    with open(sta_path, 'r') as f:
        content = f.read()
    return 'THE ANALYSIS HAS COMPLETED SUCCESSFULLY' in content

def job_has_failed(job_name, job_dir):
    """Check .sta for an explicit failure message."""
    sta_path = os.path.join(job_dir, job_name + '.sta')
    if not os.path.exists(sta_path):
        return False
    with open(sta_path, 'r') as f:
        content = f.read()
    return 'THE ANALYSIS HAS NOT BEEN COMPLETED' in content

def wait_for_slot(submitted_jobs):
    while True:
        running = sum(
            1 for jn, jd in submitted_jobs
            if not job_is_complete(jn, jd) and not job_has_failed(jn, jd)
        )
        if running < MAX_PARALLEL:
            break
        print('>> Max parallel jobs ({}) reached. Waiting {}s...'.format(
            MAX_PARALLEL, POLL_INTERVAL))
        time.sleep(POLL_INTERVAL)

def wait_for_all(submitted_jobs):
    print('>> Waiting for all jobs to finish...')
    while True:
        all_done = all(
            job_is_complete(jn, jd) or job_has_failed(jn, jd)
            for jn, jd in submitted_jobs
        )
        if all_done:
            break
        time.sleep(POLL_INTERVAL)
    print('>> All jobs finished.')

# ============================================================
# SETUP
# ============================================================

m = mdb.models['Model-1']
a = m.rootAssembly

# --- Rigid Plate (Bottom) ---
m.ConstrainedSketch(name='__profile__', sheetSize=200.0)
m.sketches['__profile__'].CircleByCenterPerimeter(
    center=(0.0, 0.0), point1=(-13.75, -7.5))
m.sketches['__profile__'].RadialDimension(
    curve=m.sketches['__profile__'].geometry[2],
    radius=25.0, textPoint=(-21.116, -12.805))
m.Part(dimensionality=THREE_D, name='Bottom', type=DISCRETE_RIGID_SURFACE)
m.parts['Bottom'].BaseSolidExtrude(depth=5.0, sketch=m.sketches['__profile__'])
del m.sketches['__profile__']
m.parts['Bottom'].RemoveCells(cellList=
    m.parts['Bottom'].cells.getSequenceFromMask(mask=('[#1 ]',)))
m.parts['Bottom'].ReferencePoint(point=
    m.parts['Bottom'].InterestingPoint(m.parts['Bottom'].edges[0], CENTER))
m.parts['Bottom'].engineeringFeatures.PointMassInertia(
    alpha=0.0, composite=0.0, i11=1.0, i22=0.0243, i33=1.0, mass=7.76e-05,
    name='Inertia-1', region=Region(referencePoints=(
        m.parts['Bottom'].referencePoints[3],)))

m.Part(name='Top', objectToCopy=m.parts['Bottom'])

# --- Material ---
m.Material(name='ABS')
m.materials['ABS'].Density(table=((1.04e-09,),))
m.materials['ABS'].Elastic(table=((1800.0, 0.35),))
m.materials['ABS'].Plastic(scaleStress=None, table=(
    (40.0, 0.0), (42.0, 0.005), (44.0, 0.01), (47.0, 0.02),
    (50.0, 0.04), (52.0, 0.06), (55.0, 0.1),
    (57.0, 0.15), (58.0, 0.2),  (59.0, 0.3)))

m.HomogeneousSolidSection(material='ABS', name='ABS', thickness=None)

# --- Assembly: plates ---
a.DatumCsysByDefault(CARTESIAN)
a.Instance(dependent=ON, name='Bottom-1', part=m.parts['Bottom'])
a.Instance(dependent=ON, name='Top-1',    part=m.parts['Top'])
a.rotate(angle=-90.0, axisDirection=(10.0,0.0,0.0),
         axisPoint=(0.0,0.0,0.0), instanceList=('Bottom-1',))
a.rotate(angle=90.0,  axisDirection=(10.0,0.0,0.0),
         axisPoint=(0.0,0.0,0.0), instanceList=('Top-1',))

# --- Import first specimen for plate positioning ---
first_stp_file  = STP_FILES[0]
first_part_name = os.path.splitext(first_stp_file)[0]
mdb.openStep(os.path.join(STP_DIR, first_stp_file), scaleFromFile=OFF)
m.PartFromGeometryFile(
    combine=False, dimensionality=THREE_D,
    geometryFile=mdb.acis, name=first_part_name, type=DEFORMABLE_BODY)

y_coords = [v.pointOn[0][1] for v in m.parts[first_part_name].vertices]
if not y_coords:
    raise RuntimeError('No vertices found in first specimen: {}'.format(first_stp_file))
y_min = min(y_coords)
y_max = max(y_coords)
plate_thickness = 5.0

a.translate(instanceList=('Bottom-1',), vector=(0.0, y_min - plate_thickness, 0.0))
a.translate(instanceList=('Top-1',),    vector=(0.0, y_max + plate_thickness, 0.0))

# --- Instance first specimen into assembly now (loop assumes this is done) ---
a.Instance(dependent=ON, name=first_part_name + '-1', part=m.parts[first_part_name])

# --- Step ---
m.ExplicitDynamicsStep(improvedDtMethod=ON, name='Step-1', previous='Initial')
m.steps['Step-1'].setValues(improvedDtMethod=ON,
    massScaling=((SEMI_AUTOMATIC, MODEL, AT_BEGINNING, 0.0, 1e-05, BELOW_MIN,
    0, 0, 0.0, 0.0, 0, None),))

# --- FD Set & History Output ---
a.Set(name='FD', referencePoints=(
    a.instances['Top-1'].referencePoints[3],))
m.historyOutputRequests['H-Output-1'].setValues(
    rebar=EXCLUDE, region=m.rootAssembly.sets['FD'],
    sectionPoints=DEFAULT, variables=('U2', 'RF2'))

# --- Contact ---
m.ContactProperty('IntProp-1')
m.interactionProperties['IntProp-1'].TangentialBehavior(
    dependencies=0, directionality=ISOTROPIC, elasticSlipStiffness=None,
    formulation=PENALTY, fraction=0.005, maximumElasticSlip=FRACTION,
    pressureDependency=OFF, shearStressLimit=None, slipRateDependency=OFF,
    table=((0.3,),), temperatureDependency=OFF)
m.interactionProperties['IntProp-1'].NormalBehavior(
    allowSeparation=ON, constraintEnforcementMethod=DEFAULT,
    pressureOverclosure=HARD)
m.ContactExp(createStepName='Step-1', name='Int-1')
m.interactions['Int-1'].includedPairs.setValuesInStep(
    stepName='Step-1', useAllstar=ON)
m.interactions['Int-1'].contactPropertyAssignments.appendInStep(
    assignments=((GLOBAL, SELF, 'IntProp-1'),), stepName='Step-1')

# --- Boundary Conditions ---
m.EncastreBC(createStepName='Initial', localCsys=None, name='ConstrainBot',
    region=Region(referencePoints=(
        a.instances['Bottom-1'].referencePoints[3],)))
m.DisplacementBC(amplitude=UNSET, createStepName='Initial',
    distributionType=UNIFORM, fieldName='', localCsys=None, name='ConstrainTop',
    region=Region(referencePoints=(a.instances['Top-1'].referencePoints[3],)),
    u1=SET, u2=UNSET, u3=SET, ur1=SET, ur2=UNSET, ur3=SET)
m.SmoothStepAmplitude(data=((0.0,0.0),(1.0,1.0)), name='Amp-1', timeSpan=STEP)
m.DisplacementBC(amplitude='Amp-1', createStepName='Step-1',
    distributionType=UNIFORM, fieldName='', fixed=OFF, localCsys=None,
    name='Displacement',
    region=Region(referencePoints=(a.instances['Top-1'].referencePoints[3],)),
    u1=UNSET, u2=-5.0, u3=UNSET, ur1=UNSET, ur2=UNSET, ur3=UNSET)

# --- Mesh plates ---
m.parts['Bottom'].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=10.0)
m.parts['Bottom'].generateMesh()
m.parts['Top'].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=10.0)
m.parts['Top'].generateMesh()

print('>> Base model setup complete.')

# ============================================================
# SUBMISSION LOOP
# ============================================================

prev_part_name = first_part_name   # FIX: seed with the already-imported specimen
submitted_jobs = []

for stp_file in STP_FILES:

    part_name = os.path.splitext(stp_file)[0]
    job_name  = part_name
    job_dir   = os.path.join(SCRIPT_DIR, part_name)
    os.makedirs(job_dir, exist_ok=True)        # FIX: don't crash if dir exists
    stp_path  = os.path.join(STP_DIR, stp_file)

    print('\n=== Setting up: {} ==='.format(stp_file))

    # --- Specimen swap ---
    if prev_part_name == part_name:
        pass  # first iteration: already imported and instanced above
    else:
        del a.instances[prev_part_name + '-1']
        del m.parts[prev_part_name]
        mdb.openStep(stp_path, scaleFromFile=OFF)
        m.PartFromGeometryFile(
            combine=False, dimensionality=THREE_D,
            geometryFile=mdb.acis, name=part_name, type=DEFORMABLE_BODY)
        a.Instance(dependent=ON, name=part_name + '-1', part=m.parts[part_name])

    # --- Section + Mesh ---
    m.parts[part_name].SectionAssignment(
        offset=0.0, offsetField='', offsetType=MIDDLE_SURFACE,
        region=Region(cells=
            m.parts[part_name].cells.getSequenceFromMask(mask=('[#1 ]',))),
        sectionName='ABS', thicknessAssignment=FROM_SECTION)
    m.parts[part_name].setMeshControls(
        elemShape=TET, technique=FREE,
        regions=m.parts[part_name].cells.getSequenceFromMask(('[#1 ]',),))
    m.parts[part_name].setElementType(
        elemTypes=(
            ElemType(elemCode=UNKNOWN_HEX,   elemLibrary=EXPLICIT),
            ElemType(elemCode=UNKNOWN_WEDGE, elemLibrary=EXPLICIT),
            ElemType(elemCode=C3D10M,        elemLibrary=EXPLICIT)),
        regions=(m.parts[part_name].cells.getSequenceFromMask(('[#1 ]',),),))
    m.parts[part_name].seedPart(deviationFactor=0.1, minSizeFactor=0.1, size=10.0)
    m.parts[part_name].generateMesh()

    a.regenerate()

    # --- Create job ---
    mdb.Job(
        activateLoadBalancing=False, atTime=None, contactPrint=OFF,
        description='', echoPrint=OFF, explicitPrecision=SINGLE,
        historyPrint=OFF, memory=90, memoryUnits=PERCENTAGE,
        model='Model-1', modelPrint=OFF, multiprocessingMode=DEFAULT,
        name=job_name, nodalOutputPrecision=SINGLE,
        numCpus=CPUS_PER_JOB, numDomains=CPUS_PER_JOB,
        numThreadsPerMpiProcess=1, queue=None, resultsFormat=ODB,
        scratch=job_dir, type=ANALYSIS, userSubroutine='',
        waitHours=0, waitMinutes=0)

    wait_for_slot(submitted_jobs)

    original_dir = os.getcwd()          # FIX: save cwd before changing it
    os.chdir(job_dir)
    mdb.jobs[job_name].submit(consistencyChecking=OFF)
    os.chdir(original_dir)              # FIX: restore immediately after submit
    print('>> Submitted: {}'.format(job_name))

    submitted_jobs.append((job_name, job_dir))
    prev_part_name = part_name

# ============================================================
# POST-PROCESSING PASS
# ============================================================

wait_for_all(submitted_jobs)

print('\n=== Post-processing ===')

os.makedirs(OUTPUT_DIR, exist_ok=True)  # ensure output dir exists

for job_name, job_dir in submitted_jobs:

    # --- Skip failed jobs gracefully ---
    if job_has_failed(job_name, job_dir):
        print('>> SKIPPING failed job: {}'.format(job_name))
        continue

    odb_path = os.path.join(job_dir, job_name + '.odb')
    if not os.path.exists(odb_path):
        print('>> WARNING: ODB not found for {}, skipping.'.format(job_name))
        continue

    try:
        odb  = openOdb(path=odb_path, readOnly=True)
        step = odb.steps['Step-1']

        fd_region_key = None
        for key in step.historyRegions.keys():
            if 'TOP-1' in key.upper():
                fd_region_key = key
                break

        if fd_region_key is None:
            print('>> WARNING: FD region not found for {}. Keys:'.format(job_name))
            for key in step.historyRegions.keys():
                print('   ' + key)
        else:
            hr  = step.historyRegions[fd_region_key]
            u2  = hr.historyOutputs['U2'].data
            rf2 = hr.historyOutputs['RF2'].data

            # FIX: warn if data lengths are mismatched
            if len(u2) != len(rf2):
                print('>> WARNING: U2 ({}) and RF2 ({}) lengths differ for {}'.format(
                    len(u2), len(rf2), job_name))

            csv_path = os.path.join(OUTPUT_DIR, job_name + '_FD.csv')
            with open(csv_path, 'w') as f:
                f.write('Time,Displacement_U2_mm,ReactionForce_RF2_N\n')
                for (t, u), (_, r) in zip(u2, rf2):
                    f.write('{},{},{}\n'.format(t, -u, -r))
            print('>> FD data saved: {}'.format(csv_path))

    except Exception as e:
        print('>> ERROR during post-processing of {}: {}'.format(job_name, str(e)))

    finally:
        try:
            odb.close()
        except Exception:
            pass

    # --- Cleanup ---
    for ext in ['.abq', '.mdl', '.prt']:
        f_path = os.path.join(job_dir, job_name + ext)
        if os.path.exists(f_path):
            os.remove(f_path)
            print('>> Deleted: {}'.format(f_path))

print('\n=== Pipeline complete! {} simulations run. ==='.format(len(submitted_jobs)))