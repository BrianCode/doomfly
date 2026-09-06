'use client';
import {useEffect,useState} from 'react';
import type {Live} from './live';
import {BroadcastPlayer} from './broadcast-player';
import {createBroadcastSession} from './broadcast-session';

export function useBroadcast(){
 const [data,setData]=useState<Live|null>(null);
 const [error,setError]=useState('Connecting to the broadcast');
 const [now,setNow]=useState(0);
 useEffect(()=>{
  const player=new BroadcastPlayer();
  const session=createBroadcastSession(frames=>player.ingest(frames),setError);
  let painting=false,lastClock=0;
  const pause=()=>{painting=false;session.pause();player.discardPending();};
  const resume=()=>{
   if(document.hidden){pause();return;}
   if(!navigator.onLine){pause();setNow(Date.now());setError('You’re offline. Reconnecting when your connection returns.');return;}
   painting=true;setNow(Date.now());session.resume();
  };
  const paint=setInterval(()=>{
   if(!painting||document.hidden)return;
   const time=Date.now();const next=player.take(time);
   if(next)setData(next);
   if(time-lastClock>=500){setNow(time);lastClock=time;}
  },50);
  document.addEventListener('visibilitychange',resume);
  window.addEventListener('pageshow',resume);
  window.addEventListener('pagehide',pause);
  window.addEventListener('online',resume);
  window.addEventListener('offline',resume);
  resume();
  return()=>{
   session.stop();clearInterval(paint);
   document.removeEventListener('visibilitychange',resume);
   window.removeEventListener('pageshow',resume);
   window.removeEventListener('pagehide',pause);
   window.removeEventListener('online',resume);
   window.removeEventListener('offline',resume);
  };
 },[]);
 return {data,error,now};
}
