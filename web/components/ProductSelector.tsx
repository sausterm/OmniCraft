'use client';

import { useState, useEffect } from 'react';
import { Check, ShoppingCart, Loader2 } from 'lucide-react';
import type { Product } from '@/lib/api';

interface ProductSelectorProps {
  products: Product[];
  onCheckout: (selectedProductIds: string[]) => void;
  isLoading?: boolean;
}

export default function ProductSelector({
  products,
  onCheckout,
  isLoading = false,
}: ProductSelectorProps) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Auto-select free products
  useEffect(() => {
    const freeIds = products
      .filter((p) => p.price === 0 && !p.purchased)
      .map((p) => p.id);
    if (freeIds.length > 0) {
      setSelectedIds(new Set(freeIds));
    }
  }, [products]);

  const toggleProduct = (productId: string) => {
    const product = products.find((p) => p.id === productId);
    if (!product || product.purchased) return;

    const newSelected = new Set(selectedIds);
    if (newSelected.has(productId)) {
      newSelected.delete(productId);
    } else {
      newSelected.add(productId);
    }
    setSelectedIds(newSelected);
  };

  const selectedProducts = products.filter((p) => selectedIds.has(p.id));
  const totalPrice = selectedProducts.reduce((sum, p) => sum + p.price, 0);

  const formatPrice = (cents: number) => {
    if (cents === 0) return 'Free';
    return `$${(cents / 100).toFixed(2)}`;
  };

  const handleCheckout = () => {
    onCheckout(Array.from(selectedIds));
  };

  // Separate tiers from add-ons
  const tiers = products.filter((p) =>
    ['preview', 'basic', 'standard', 'premium'].includes(p.id)
  );
  const addons = products.filter(
    (p) => !['preview', 'basic', 'standard', 'premium'].includes(p.id)
  );

  return (
    <div className="space-y-6">
      {/* Main Tiers */}
      <div>
        <h3 className="font-medium text-gray-900 mb-3">Choose Your Package</h3>
        <div className="grid sm:grid-cols-2 gap-3">
          {tiers.map((product) => {
            const isSelected = selectedIds.has(product.id);
            const isPurchased = product.purchased;

            return (
              <button
                key={product.id}
                onClick={() => toggleProduct(product.id)}
                disabled={isPurchased || isLoading}
                className={`p-4 rounded-xl border-2 text-left transition-all ${
                  isPurchased
                    ? 'border-accent-500 bg-accent-50 cursor-default'
                    : isSelected
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                } disabled:cursor-not-allowed`}
              >
                <div className="flex items-start justify-between mb-2">
                  <div>
                    <span className="font-semibold text-gray-900">
                      {product.name}
                    </span>
                    <span className="ml-2 text-lg font-bold text-primary-600">
                      {formatPrice(product.price)}
                    </span>
                  </div>
                  {isPurchased ? (
                    <span className="flex items-center gap-1 text-xs text-accent-600 bg-accent-100 px-2 py-1 rounded-full">
                      <Check className="w-3 h-3" />
                      Purchased
                    </span>
                  ) : isSelected ? (
                    <div className="w-5 h-5 bg-primary-500 rounded-full flex items-center justify-center">
                      <Check className="w-3 h-3 text-white" />
                    </div>
                  ) : (
                    <div className="w-5 h-5 border-2 border-gray-300 rounded-full" />
                  )}
                </div>
                <p className="text-sm text-gray-600 mb-2">{product.description}</p>
                <ul className="space-y-1">
                  {product.includes.map((item) => (
                    <li
                      key={item}
                      className="text-xs text-gray-500 flex items-center gap-1"
                    >
                      <Check className="w-3 h-3 text-accent-500" />
                      {item.replace(/_/g, ' ')}
                    </li>
                  ))}
                </ul>
              </button>
            );
          })}
        </div>
      </div>

      {/* Add-ons */}
      {addons.length > 0 && (
        <div>
          <h3 className="font-medium text-gray-900 mb-3">Optional Add-ons</h3>
          <div className="space-y-2">
            {addons.map((product) => {
              const isSelected = selectedIds.has(product.id);
              const isPurchased = product.purchased;

              return (
                <button
                  key={product.id}
                  onClick={() => toggleProduct(product.id)}
                  disabled={isPurchased || isLoading}
                  className={`w-full p-3 rounded-lg border text-left transition-all flex items-center justify-between ${
                    isPurchased
                      ? 'border-accent-500 bg-accent-50'
                      : isSelected
                      ? 'border-primary-500 bg-primary-50'
                      : 'border-gray-200 hover:border-gray-300'
                  } disabled:cursor-not-allowed`}
                >
                  <div className="flex items-center gap-3">
                    {isPurchased ? (
                      <span className="w-5 h-5 bg-accent-500 rounded flex items-center justify-center">
                        <Check className="w-3 h-3 text-white" />
                      </span>
                    ) : isSelected ? (
                      <span className="w-5 h-5 bg-primary-500 rounded flex items-center justify-center">
                        <Check className="w-3 h-3 text-white" />
                      </span>
                    ) : (
                      <span className="w-5 h-5 border-2 border-gray-300 rounded" />
                    )}
                    <div>
                      <span className="font-medium text-gray-900">
                        {product.name}
                      </span>
                      <span className="text-gray-500 text-sm ml-2">
                        {product.description}
                      </span>
                    </div>
                  </div>
                  <span className="font-semibold text-primary-600">
                    {isPurchased ? 'Owned' : formatPrice(product.price)}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Checkout Button */}
      <div className="border-t border-gray-200 pt-4">
        <div className="flex items-center justify-between mb-4">
          <span className="text-gray-600">Total</span>
          <span className="text-2xl font-bold text-gray-900">
            {formatPrice(totalPrice)}
          </span>
        </div>
        <button
          onClick={handleCheckout}
          disabled={selectedIds.size === 0 || isLoading}
          className="btn-primary w-full py-3 flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isLoading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              Processing...
            </>
          ) : totalPrice === 0 ? (
            <>
              <Check className="w-5 h-5" />
              Get Free Preview
            </>
          ) : (
            <>
              <ShoppingCart className="w-5 h-5" />
              Proceed to Checkout
            </>
          )}
        </button>
      </div>
    </div>
  );
}
