import {PixelIcon} from '@/components/pixel-icon';

export function SiteHeader({page}:{page:'home'|'how-it-works'}){
 const BrandTitle=page==='home'?'h1':'span';
 return <header className="masthead site-header">
  <a href="/" className="brand" aria-label="DOOMFLY home">
   <span className="brand-icon"><img src="/fly-logo.svg?v=side-fly" alt="" width="36" height="36"/></span>
   <BrandTitle className="brand-title">DOOMFLY</BrandTitle>
  </a>
  <nav className="header-actions" aria-label="Main navigation">
   <a className="header-guide" href="/how-it-works" aria-current={page==='how-it-works'?'page':undefined}>How it works</a>
   <a className="icon-link" href="https://github.com/nftechie/doomfly" target="_blank" rel="noopener noreferrer" aria-label="View simulation source on GitHub (opens in a new tab)" title="View source on GitHub"><PixelIcon name="code"/></a>
  </nav>
 </header>;
}
