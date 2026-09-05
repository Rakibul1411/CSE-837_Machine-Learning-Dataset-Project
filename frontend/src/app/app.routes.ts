import { Routes } from '@angular/router';
import { DashboardComponent } from './components/dashboard/dashboard.component';
import { ForecastComponent } from './components/forecast/forecast.component';
import { PredictComponent } from './components/predict/predict.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'dashboard' },
  { path: 'dashboard', component: DashboardComponent },
  { path: 'train', redirectTo: 'dashboard' },
  { path: 'predict', component: PredictComponent },
  { path: 'forecast', component: ForecastComponent },
];
