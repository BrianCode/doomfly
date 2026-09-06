import {execFileSync} from 'node:child_process';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
const out=mkdtempSync(join(tmpdir(),'doomfly-web-tests-'));
try{
 execFileSync('node_modules/.bin/tsc',['--outDir',out,'--module','commonjs','--target','ES2022','--skipLibCheck','--esModuleInterop','--types','node','lib/live.ts','lib/broadcast-player.ts','lib/broadcast-proxy.ts','lib/broadcast-session.ts'],{stdio:'inherit'});
 execFileSync(process.execPath,['--test','tests/broadcast.test.cjs','tests/broadcast-session.test.cjs'],{stdio:'inherit',env:{...process.env,DOOMFLY_TEST_BUILD:out}});
}finally{rmSync(out,{recursive:true,force:true});}
