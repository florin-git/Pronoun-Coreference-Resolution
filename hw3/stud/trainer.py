import os
import time
import math

from typing import *
from datetime import datetime
from arguments import CustomTrainingArguments

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler


class GeneralTrainer:    
    def __init__(
        self,
        device: str,
        model: nn.Module,
        args: CustomTrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        assert args is not None, "No training arguments passed!"
        self.args = args
        
    def train(self):
        args = self.args
        valid_dataloader = self.valid_dataloader
        epochs = args.num_train_epochs
        
        train_losses = []
        train_acc_list = []
        valid_losses = []
        valid_acc_list = []
        
        if args.use_early_stopping:
            patience_counter = 0 

        scaler = GradScaler() if args.use_scaler else None

        training_start_time = time.time()
        print("\nTraining...")
        for epoch in range(epochs):
            train_loss, train_acc = self._inner_training_loop(scaler)
            train_losses.append(train_loss)
            train_acc_list.append(train_acc)

            valid_loss, valid_acc = self.evaluate(valid_dataloader)
            valid_losses.append(valid_loss)
            valid_acc_list.append(valid_acc)

            if self.scheduler is not None:
                self._print_sceduler_lr()
                self.scheduler.step()

            self._print_epoch_log(epoch, epochs, train_loss, valid_loss, valid_acc)

            if args.use_early_stopping and len(valid_acc_list) >= 2:
                stop, patience_counter = self._early_stopping(patience_counter, epoch, valid_acc_list)
                if stop:
                    break
        
        training_time = time.time() - training_start_time
        print(f'Training time: {self._print_time(training_time)}')

        metrics_history = {
            "train_losses": train_losses,
            "train_acc": train_acc_list,
            "valid_losses": valid_losses,
            "valid_acc": valid_acc_list,
        }
        print(metrics_history)
        if args.save_model:
            self._save_model(args.task_type, epoch, valid_acc, scaler, metrics_history)
    
        return metrics_history

    def _inner_training_loop(self, scaler):
        pass

    def evaluate(self, eval_dataloader):
        pass
    
    def _early_stopping(self, patience_counter, epoch, valid_acc_list):
        args = self.args

        # stop = args.early_stopping_mode == 'min' and epoch > 0 and valid_acc_list[-1] > valid_acc_list[-2]
        stop = args.early_stopping_mode == 'max' and epoch > 0 and valid_acc_list[-1] < valid_acc_list[-2]
        if stop:
            if patience_counter >= args.early_stopping_patience:
                print('Early stop.')
                return stop, patience_counter
            else:
                print('-- Patience.\n')
                patience_counter += 1

        return False, patience_counter   
    
    def _print_time(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def _print_sceduler_lr(self):
        print('-' * 17)
        print(f"| LR: {self.scheduler.get_last_lr()[0]:.3e} |")

    def _print_step_log(self, step, running_loss, running_acc):
        print(f'\t| step {step+1:4d}/{len(self.train_dataloader):d} | train_loss: {running_loss:.3f} | ' \
                f'train_acc: {running_acc:.3f} |')

    def _print_epoch_log(self, epoch, epochs, train_loss, valid_loss, valid_acc):
        print('-' * 76)
        print(f'| epoch {epoch+1:>3d}/{epochs:<3d} | train_loss: {train_loss:.3f} | ' \
                f'valid_loss: {valid_loss:.3f} | valid_acc: {valid_acc:.3f} |')
        print('-' * 76)
        
    
    def _save_model(self, task_type, epoch, valid_acc, scaler, metrics_history):
        print("Saving model...")
        params_to_save = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics_history": metrics_history,
        }
        
        if self.scheduler is not None:
            params_to_save["scheduler_state_dict"] = self.scheduler.state_dict()
            
        if scaler is not None:
            params_to_save["scaler_state_dict"] = scaler.state_dict()
            
        save_path = f"{self.args.output_dir}my_model{str(task_type)}_{str(valid_acc)[2:5]}_{epoch+1}"
        now = datetime.now()
        current_time = now.strftime("%H-%M-%S")
        
        if os.path.exists(f"{save_path}_{current_time}.pth"):
            torch.save(params_to_save, f"{save_path}_{current_time}_new.pth")
        else:
            torch.save(params_to_save, f"{save_path}_{current_time}.pth")
        
        print("Model saved.")


class Trainer(GeneralTrainer):
    def __init__(
        self,
        device: str,
        model: nn.Module,
        args: CustomTrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        super(Trainer, self).__init__(
            device, 
            model, 
            args, 
            train_dataloader,
            valid_dataloader, 
            criterion, 
            optimizer, 
            scheduler, 
        )


    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_correct, total_count = 0.0, 0.0

        self.model.train()
        for step, sample in enumerate(train_dataloader):
            ### Empty gradients ###
            self.optimizer.zero_grad(set_to_none=True)
            
            ### Forward ###
            if scaler is None:
                predictions = self.model(sample)
                labels = sample['labels']
                loss = self.criterion(predictions, labels)
            
            else:
                with torch.autocast(device_type=self.device):
                    predictions = self.model(sample)
                    labels = sample['labels']
                    loss = self.criterion(predictions, labels)

            train_correct += (predictions.argmax(1) == labels).sum().item()
            total_count += labels.shape[0]
            
            ### Backward  ###
            if scaler is None:
                loss.backward()
            else: 
                # Backward pass without mixed precision
                # It's not recommended to use mixed precision for backward pass
                # Because we need more precise loss
                scaler.scale(loss).backward()
            
            if args.grad_clipping is not None:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clipping)
            
            ### Update weights ### 
            if scaler is None:
                self.optimizer.step()
            else:
                scaler.step(self.optimizer)
                scaler.update()

            train_loss += loss.item()

            if step % args.logging_steps == args.logging_steps - 1:
                running_loss = train_loss / (step + 1)
                running_acc = train_correct / total_count
                self._print_step_log(step, running_loss, running_acc)
                
        return train_loss / len(train_dataloader), train_correct / total_count


    def evaluate(self, eval_dataloader):
        valid_loss = 0.0
        eval_correct, total_count = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for sample in eval_dataloader:
                
                predictions = self.model(sample)
                labels = sample['labels']
                
                loss = self.criterion(predictions, labels)
                valid_loss += loss.item()

                eval_correct += (predictions.argmax(1) == labels).sum().item()
                total_count += labels.shape[0]
        
        return valid_loss / len(eval_dataloader), eval_correct / total_count

class TokenClassificationTrainer(GeneralTrainer):
    def __init__(
        self,
        device: str,
        model: nn.Module,
        args: CustomTrainingArguments,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        criterion: torch.nn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler = None,
        pad: int = 0,
    ):
        
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.pad = pad

        super(TokenClassificationTrainer, self).__init__(
            device, 
            model, 
            args, 
            train_dataloader,
            valid_dataloader, 
            criterion, 
            optimizer, 
            scheduler, 
        )

    def _inner_training_loop(self, scaler):
        args = self.args
        train_dataloader = self.train_dataloader
        
        train_loss = 0.0
        train_correct, total_count = 0.0, 0.0

        self.model.train()
        for step, sample in enumerate(train_dataloader):
            ### Empty gradients ###
            self.optimizer.zero_grad(set_to_none=True)
            
            ### Forward ###
            if scaler is None:
                predictions = self.model(sample)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = sample['labels']
                labels = labels.view(-1)
                loss = self.criterion(predictions, labels)
            
            else:
                with torch.autocast(device_type=self.device):
                    predictions = self.model(sample)
                    predictions = predictions.view(-1, predictions.shape[-1])
                    labels = sample['labels']
                    labels = labels.view(-1)
                    loss = self.criterion(predictions, labels)

            mask = labels != self.pad
            predictions = predictions.argmax(1)
            predictions = predictions[mask]
            labels = labels[mask]
            train_correct += (predictions == labels).sum().item()
            total_count += labels.shape[0]
            
            ### Backward  ###
            if scaler is None:
                loss.backward()
            else: 
                # Backward pass without mixed precision
                # It's not recommended to use mixed precision for backward pass
                # Because we need more precise loss
                scaler.scale(loss).backward()
            
            if args.grad_clipping is not None:
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.grad_clipping)
            
            ### Update weights ### 
            if scaler is None:
                self.optimizer.step()
            else:
                scaler.step(self.optimizer)
                scaler.update()

            train_loss += loss.item()

            if step % args.logging_steps == args.logging_steps - 1:
                running_loss = train_loss / (step + 1)
                running_acc = train_correct / total_count
                self._print_step_log(step, running_loss, running_acc)
                
        return train_loss / len(train_dataloader), train_correct / total_count


    def evaluate(self, eval_dataloader):
        valid_loss = 0.0
        eval_correct, total_count = 0, 0
        
        self.model.eval()
        with torch.no_grad():
            for sample in eval_dataloader:
                
                predictions = self.model(sample)
                predictions = predictions.view(-1, predictions.shape[-1])
                labels = sample['labels']
                labels = labels.view(-1)
                loss = self.criterion(predictions, labels)
                valid_loss += loss.item()
                
                mask = labels != self.pad
                predictions = predictions.argmax(1)
                predictions = predictions[mask]
                labels = labels[mask]
                eval_correct += (predictions == labels).sum().item()
                total_count += labels.shape[0]
        
        return valid_loss / len(eval_dataloader), eval_correct / total_count