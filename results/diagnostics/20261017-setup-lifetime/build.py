from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys, time
root=Path(__file__).resolve().parents[3]
diag=Path(__file__).resolve().parent
commands=[]
def run(command,cwd,name):
    print(name, flush=True)
    with (diag/(name+'.stdout.log')).open('wb') as out, (diag/(name+'.stderr.log')).open('wb') as err:
        completed=subprocess.run(command,cwd=cwd,stdout=out,stderr=err,timeout=600)
    commands.append({'name':name,'command':command,'cwd':str(cwd),'exit_code':completed.returncode})
    (diag/'build-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    if completed.returncode:raise SystemExit(f'{name} failed: see {diag/(name+".stderr.log")}')
for crate in ('undine','borg'):
    target=diag/('build-'+crate)
    base=['cargo','--locked','--release','--features','bench','--target-dir',str(target)]
    run(['cargo','test',*base[1:],'--test','benchmark_setup'],root/crate,crate+'-setup-tests')
    names=['bench_append_only_log_client_point_open_local','bench_append_only_log_point_open_local']
    if crate=='undine':names.append('bench_append_only_log_on_the_fly_open_local')
    args=['cargo','build',*base[1:],'--message-format=json']
    for name in names:args+=['--bench',name]
    run(args,root/crate,crate+'-point-build')
    destination=diag/'binaries'/crate
    destination.mkdir(parents=True,exist_ok=True)
    for name in names:
        messages=[json.loads(line) for line in (diag/(crate+'-point-build.stdout.log')).read_text().splitlines() if line.startswith('{')]
        executable=next(m['executable'] for m in messages if m.get('reason')=='compiler-artifact' and m.get('target',{}).get('name')==name and m.get('executable'))
        shutil.copy2(executable,destination/name)
    name='bench_append_only_log_range_open_local'
    run(['cargo','build','--locked','--release','--features','bench,log-bulk-updates-8mb','--target-dir',str(target),'--bench',name,'--message-format=json'],root/crate,crate+'-range-build')
    messages=[json.loads(line) for line in (diag/(crate+'-range-build.stdout.log')).read_text().splitlines() if line.startswith('{')]
    executable=next(m['executable'] for m in messages if m.get('reason')=='compiler-artifact' and m.get('target',{}).get('name')==name and m.get('executable'))
    shutil.copy2(executable,destination/name)
print('All builds and ownership tests passed',flush=True)
