import type {Metadata} from 'next';
import LearningLab from '@/components/learning-lab';
import LearningIterations from '@/components/learning-iterations';
import report from '@/data/learning-report.json';
import iterations from '@/data/learning-iterations.json';
import './lab.css';

export const metadata:Metadata={title:'Learning lab · DOOMFLY',description:'Full-connectome learning experiments, measured controls, and the tests the model still needs to pass.'};
// The tiny static pixel SVG is already resolution independent. Serving it
// directly also avoids the image shim's additional client runtime.
/* oxlint-disable next/no-img-element */
export default function LearningPage(){return <main className="arcade lab-page">
 <header className="masthead"><a href="/" className="brand how-brand" aria-label="DOOMFLY home"><span className="brand-icon"><img src="/fly-logo.svg?v=side-fly" width="36" height="36" alt=""/></span><span>DOOMFLY</span></a><div className="header-actions"><a className="header-guide" href="/how-it-works">How it works</a><a className="header-guide" href="/">Watch</a></div></header>
 <LearningLab report={report} latest={<LearningIterations report={iterations}/>}/>
 </main>}
